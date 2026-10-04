#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

static void die_code(int code, const char *format, ...) {
    va_list args;
    fputs("q27: ", stderr);
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fputc('\n', stderr);
    exit(code);
}

static void die_errno(const char *prefix, const char *path) {
    die_code(75, "%s %s: %s", prefix, path, strerror(errno));
}

static void make_absolute(const char *input, char output[PATH_MAX]) {
    if (!input || !*input) die_code(75, "invalid consumer lock path");
    if (input[0] == '/') {
        if (snprintf(output, PATH_MAX, "%s", input) >= PATH_MAX)
            die_code(75, "consumer lock path is too long");
        return;
    }
    if (!getcwd(output, PATH_MAX)) die_errno("cannot resolve consumer lock path from", ".");
    const size_t used = strlen(output);
    if (used + 1 + strlen(input) + 1 > PATH_MAX)
        die_code(75, "consumer lock path is too long");
    output[used] = '/';
    strcpy(output + used + 1, input);
}

static void mkdir_parents(const char *parent) {
    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s", parent) >= (int)sizeof(path))
        die_code(75, "consumer run path is too long");
    for (char *p = path + 1; *p; ++p) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(path, 0700) != 0 && errno != EEXIST)
            die_errno("cannot create consumer run directory", path);
        *p = '/';
    }
    if (mkdir(path, 0700) != 0 && errno != EEXIST)
        die_errno("cannot create consumer run directory", path);
}

int main(int argc, char **argv) {
    if (argc < 3)
        die_code(2, "usage: q27-lock-exec LOCK COMMAND [ARGS...]");

    char lock_path[PATH_MAX];
    make_absolute(argv[1], lock_path);
    char *slash = strrchr(lock_path, '/');
    if (!slash || !slash[1] || strcmp(slash + 1, ".") == 0 ||
        strcmp(slash + 1, "..") == 0)
        die_code(75, "invalid consumer lock path");
    const char *leaf = slash + 1;
    char parent[PATH_MAX];
    const size_t parent_len = slash == lock_path ? 1u : (size_t)(slash - lock_path);
    if (parent_len >= sizeof(parent)) die_code(75, "consumer run path is too long");
    memcpy(parent, lock_path, parent_len);
    parent[parent_len] = '\0';

    const mode_t old_umask = umask(0077);
    mkdir_parents(parent);
    umask(old_umask);

    struct stat parent_named;
    if (lstat(parent, &parent_named) != 0)
        die_errno("cannot protect consumer run directory", parent);
    if (!S_ISDIR(parent_named.st_mode))
        die_code(75, "consumer run path is not a directory: %s", parent);

    const int dir_fd = open(parent, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dir_fd < 0) die_errno("cannot protect consumer run directory", parent);
    struct stat parent_opened;
    if (fstat(dir_fd, &parent_opened) != 0) {
        const int saved = errno;
        close(dir_fd);
        errno = saved;
        die_errno("cannot inspect consumer run directory", parent);
    }
    if (!S_ISDIR(parent_opened.st_mode)) {
        close(dir_fd);
        die_code(75, "consumer run path is not a directory: %s", parent);
    }
    if (parent_opened.st_uid != geteuid()) {
        close(dir_fd);
        die_code(75, "consumer run directory is not owned by this user: %s", parent);
    }
    if (fchmod(dir_fd, 0700) != 0) {
        const int saved = errno;
        close(dir_fd);
        errno = saved;
        die_errno("cannot protect consumer run directory", parent);
    }
    if (fstat(dir_fd, &parent_opened) != 0 ||
        (parent_opened.st_mode & 07777) != 0700) {
        close(dir_fd);
        die_code(75, "consumer run directory is not private: %s", parent);
    }

    const int fd = openat(dir_fd, leaf,
                          O_RDWR | O_CREAT | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC,
                          0600);
    if (fd < 0) {
        const int saved = errno;
        close(dir_fd);
        errno = saved;
        die_errno("cannot launch locked consumer:", lock_path);
    }
    if (fchmod(fd, 0600) != 0) {
        const int saved = errno;
        close(fd);
        close(dir_fd);
        errno = saved;
        die_errno("cannot launch locked consumer:", lock_path);
    }

    struct stat opened, named;
    if (fstat(fd, &opened) != 0 ||
        fstatat(dir_fd, leaf, &named, AT_SYMLINK_NOFOLLOW) != 0) {
        const int saved = errno;
        close(fd);
        close(dir_fd);
        errno = saved;
        die_errno("cannot launch locked consumer:", lock_path);
    }
    if (!S_ISREG(opened.st_mode) || opened.st_uid != geteuid() ||
        opened.st_nlink != 1 || (opened.st_mode & 07777) != 0600 ||
        opened.st_dev != named.st_dev || opened.st_ino != named.st_ino) {
        close(fd);
        close(dir_fd);
        die_code(75, "consumer lock is not one private regular file");
    }

    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        const int saved = errno;
        close(fd);
        close(dir_fd);
        if (saved == EWOULDBLOCK || saved == EAGAIN)
            die_code(75, "a q27 server or native agent is already running (one model at a time)");
        errno = saved;
        die_errno("cannot launch locked consumer:", lock_path);
    }

    // Revalidate after acquisition. The private owner-only parent excludes
    // other principals; this catches accidental same-account replacement.
    if (fstatat(dir_fd, leaf, &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        opened.st_dev != named.st_dev || opened.st_ino != named.st_ino) {
        close(fd);
        close(dir_fd);
        die_code(75, "consumer lock identity changed during acquisition");
    }

    int fd_flags = fcntl(fd, F_GETFD);
    if (fd_flags < 0 || fcntl(fd, F_SETFD, fd_flags & ~FD_CLOEXEC) != 0) {
        const int saved = errno;
        close(fd);
        close(dir_fd);
        errno = saved;
        die_errno("cannot make consumer lock inheritable:", lock_path);
    }
    char fd_text[32];
    snprintf(fd_text, sizeof(fd_text), "%d", fd);
    if (setenv("Q27_SUPERVISOR_LOCK_FD", fd_text, 1) != 0) {
        const int saved = errno;
        close(fd);
        close(dir_fd);
        errno = saved;
        die_errno("cannot export consumer lock descriptor:", lock_path);
    }
    close(dir_fd);

    execvp(argv[2], &argv[2]);
    die_errno("cannot launch locked consumer:", argv[2]);
    return 75;
}
