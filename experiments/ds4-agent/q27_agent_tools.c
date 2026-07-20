#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "q27_agent_tools.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/acl.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>
#if defined(__linux__)
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/xattr.h>
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1u << 0)
#endif
#ifndef RENAME_EXCHANGE
#define RENAME_EXCHANGE (1u << 1)
#endif
#endif
#include <time.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

#define FILE_MAX_BYTES (8u * 1024u * 1024u)
#define OUTPUT_MAX_BYTES (256u * 1024u)

// Darwin extended allow ACLs can grant access independently of restrictive
// BSD mode bits. Reject a granting ACL on the destination parent before
// staging, then strip and verify ACLs on both private staging inodes.
static int fd_has_granting_acl(int fd) {
#if defined(__APPLE__)
    errno = 0;
    acl_t acl = acl_get_fd_np(fd, ACL_TYPE_EXTENDED);
    if (!acl) return errno == ENOENT ? 0 : -1;
    acl_entry_t entry;
    int entry_id = ACL_FIRST_ENTRY;
    for (;;) {
        errno = 0;
        int got = acl_get_entry(acl, entry_id, &entry);
        if (got < 0) {
            int saved = errno;
            acl_free(acl);
            return saved == EINVAL ? 0 : -1;
        }
        acl_tag_t tag;
        if (acl_get_tag_type(entry, &tag) != 0) {
            acl_free(acl);
            return -1;
        }
        if (tag == ACL_EXTENDED_ALLOW) {
            acl_free(acl);
            return 1;
        }
        if (tag != ACL_EXTENDED_DENY) {
            acl_free(acl);
            errno = EINVAL;
            return -1;
        }
        entry_id = ACL_NEXT_ENTRY;
    }
#elif defined(__linux__)
    static const char *names[] = {
        "system.posix_acl_access", "system.posix_acl_default"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        errno = 0;
        ssize_t size = fgetxattr(fd, names[i], NULL, 0);
        if (size >= 0) return 1; // reject any inherited POSIX ACL
        if (errno != ENODATA && errno != ENOTSUP && errno != ENOTDIR)
            return -1;
    }
    return 0;
#else
    (void)fd;
    return 0;
#endif
}

static int clear_extended_acl_fd(int fd) {
#if defined(__APPLE__)
    acl_t empty = acl_init(0);
    if (!empty) return 0;
    int ok = acl_set_fd_np(fd, empty, ACL_TYPE_EXTENDED) == 0;
    acl_free(empty);
    if (!ok) return 0;
    errno = 0;
    acl_t verify = acl_get_fd_np(fd, ACL_TYPE_EXTENDED);
    if (!verify) return errno == ENOENT;
    acl_entry_t entry;
    errno = 0;
    int got = acl_get_entry(verify, ACL_FIRST_ENTRY, &entry);
    int saved = errno;
    acl_free(verify);
    return got < 0 && saved == EINVAL;
#elif defined(__linux__)
    static const char *names[] = {
        "system.posix_acl_access", "system.posix_acl_default"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (fremovexattr(fd, names[i]) != 0 && errno != ENODATA &&
            errno != ENOTSUP && errno != ENOTDIR)
            return 0;
    }
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        errno = 0;
        if (fgetxattr(fd, names[i], NULL, 0) >= 0 ||
            (errno != ENODATA && errno != ENOTSUP && errno != ENOTDIR))
            return 0;
    }
    return 1;
#else
    (void)fd;
    return 1;
#endif
}

static void result_error(q27_agent_tool_result *result, const char *message) {
    result->exit_code = -1;
    snprintf(result->message, sizeof(result->message), "%s",
             message ? message : "tool failure");
}

static void result_errno(q27_agent_tool_result *result, const char *prefix) {
    char message[256];
    snprintf(message, sizeof(message), "%s: %s", prefix, strerror(errno));
    result_error(result, message);
}

static int request_valid(const q27_agent_tool_request *request,
                         q27_agent_tool_result *result) {
    if (!request || request->kind < Q27_TOOL_READ ||
        request->kind > Q27_TOOL_WRITE || !request->max_output_bytes ||
        request->max_output_bytes > OUTPUT_MAX_BYTES ||
        request->input_len > FILE_MAX_BYTES ||
        request->replacement_len > FILE_MAX_BYTES) {
        result_error(result, "invalid tool request/output bound");
        return 0;
    }
    if (request->kind == Q27_TOOL_SHELL) {
        if (!request->input || !request->input_len || !request->timeout_ms ||
            request->timeout_ms > 60000 ||
            memchr(request->input, '\0', request->input_len)) {
            result_error(result, "invalid shell command or timeout");
            return 0;
        }
        return 1;
    }
    if (!request->path || !*request->path || request->path[0] == '/') {
        result_error(result, "tool path must be non-empty and relative");
        return 0;
    }
    if (request->kind == Q27_TOOL_SEARCH &&
        (!request->input || !request->input_len)) {
        result_error(result, "search needle must not be empty");
        return 0;
    }
    if (request->kind == Q27_TOOL_WRITE &&
        ((request->input_len && !request->input) ||
         request->replacement_len)) {
        result_error(result, "invalid write content");
        return 0;
    }
    if (request->kind == Q27_TOOL_EDIT &&
        (!request->input || !request->input_len ||
         (request->replacement_len && !request->replacement))) {
        result_error(result, "edit old bytes must not be empty");
        return 0;
    }
    return 1;
}

// Opens every parent relative to rootfd with O_NOFOLLOW and returns an owned
// parent fd plus final component. Empty, dot, dot-dot, repeated, and trailing
// components are rejected rather than normalized.
static int open_parent(int rootfd, const char *path, int *parent_out,
                       char **leaf_out, q27_agent_tool_result *result) {
    const size_t path_len = strlen(path);
    if (!path_len || path_len > 4096 || path[path_len - 1] == '/') {
        result_error(result, "invalid tool path");
        return 0;
    }
    char *copy = strdup(path);
    if (!copy) { result_error(result, "out of memory"); return 0; }
    // openat(".") creates an independent open-file description. dup() would
    // share rootfd's flock state, leaking root-level edit locks for the
    // lifetime of the worker and defeating serialization between callers.
    int dir = openat(rootfd, ".",
                     O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (dir < 0) {
        result_errno(result, "cannot reopen workspace fd");
        free(copy);
        return 0;
    }
    char *component = copy;
    for (;;) {
        char *slash = strchr(component, '/');
        if (slash) *slash = '\0';
        if (!*component || !strcmp(component, ".") || !strcmp(component, "..")) {
            result_error(result, "unsafe tool path component");
            close(dir);
            free(copy);
            return 0;
        }
        if (!slash) {
            *leaf_out = strdup(component);
            if (!*leaf_out) {
                result_error(result, "out of memory");
                close(dir);
                free(copy);
                return 0;
            }
            *parent_out = dir;
            free(copy);
            return 1;
        }
        int next = openat(dir, component,
                          O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0) {
            result_errno(result, "cannot open workspace path component");
            close(dir);
            free(copy);
            return 0;
        }
        close(dir);
        dir = next;
        component = slash + 1;
    }
}

static int open_regular(int rootfd, const char *path, int flags,
                        int *parent_out, char **leaf_out, struct stat *st,
                        q27_agent_tool_result *result) {
    int parent = -1;
    char *leaf = NULL;
    if (!open_parent(rootfd, path, &parent, &leaf, result)) return -1;
    int fd = openat(parent, leaf, flags | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        result_errno(result, "cannot open tool file");
        close(parent);
        free(leaf);
        return -1;
    }
    if (fstat(fd, st) != 0) {
        result_errno(result, "cannot stat tool file");
        close(fd);
        close(parent);
        free(leaf);
        return -1;
    }
    if (!S_ISREG(st->st_mode)) {
        errno = EINVAL;
        result_errno(result, "tool target is not a regular file");
        close(fd);
        close(parent);
        free(leaf);
        return -1;
    }
    if (st->st_size < 0 || (uint64_t)st->st_size > FILE_MAX_BYTES) {
        result_error(result, "tool file exceeds 8 MiB bound");
        close(fd);
        close(parent);
        free(leaf);
        return -1;
    }
    *parent_out = parent;
    *leaf_out = leaf;
    return fd;
}

static unsigned char *read_all(int fd, size_t size,
                               q27_agent_tool_result *result) {
    unsigned char *data = malloc(size ? size : 1);
    if (!data) { result_error(result, "out of memory"); return NULL; }
    size_t offset = 0;
    while (offset < size) {
        ssize_t n = pread(fd, data + offset, size - offset, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0) errno = EIO;
            result_errno(result, "cannot read tool file");
            free(data);
            return NULL;
        }
        offset += (size_t)n;
    }
    return data;
}

static size_t *kmp_table(const unsigned char *needle, size_t nlen,
                         q27_agent_alive_check alive, void *opaque,
                         int *cancelled) {
    if (nlen > SIZE_MAX / sizeof(size_t)) return NULL;
    size_t *table = calloc(nlen, sizeof(*table));
    if (!table) return NULL;
    size_t matched = 0;
    for (size_t i = 1; i < nlen; ++i) {
        if ((i & 65535u) == 0 && !alive(opaque)) {
            *cancelled = 1;
            free(table);
            return NULL;
        }
        while (matched && needle[i] != needle[matched])
            matched = table[matched - 1];
        if (needle[i] == needle[matched]) ++matched;
        table[i] = matched;
    }
    return table;
}

static int kmp_contains(const unsigned char *haystack, size_t hlen,
                        const unsigned char *needle, size_t nlen,
                        const size_t *table, q27_agent_alive_check alive,
                        void *opaque, int *cancelled) {
    size_t matched = 0;
    for (size_t i = 0; i < hlen; ++i) {
        if ((i & 65535u) == 0 && !alive(opaque)) {
            *cancelled = 1;
            return 0;
        }
        while (matched && haystack[i] != needle[matched])
            matched = table[matched - 1];
        if (haystack[i] == needle[matched]) ++matched;
        if (matched == nlen) return 1;
    }
    return 0;
}

static int kmp_matches(const unsigned char *haystack, size_t hlen,
                       const unsigned char *needle, size_t nlen,
                       const size_t *table, q27_agent_alive_check alive,
                       void *opaque, size_t *matches, size_t *last,
                       int *cancelled) {
    size_t matched = 0;
    *matches = 0;
    *last = 0;
    for (size_t i = 0; i < hlen; ++i) {
        if ((i & 65535u) == 0 && !alive(opaque)) {
            *cancelled = 1;
            return 0;
        }
        while (matched && haystack[i] != needle[matched])
            matched = table[matched - 1];
        if (haystack[i] == needle[matched]) ++matched;
        if (matched == nlen) {
            ++*matches;
            *last = i + 1 - nlen;
            matched = table[matched - 1];
        }
    }
    return 1;
}

static q27_agent_status run_read(int rootfd,
                                 const q27_agent_tool_request *request,
                                 q27_agent_tool_sink sink,
                                 q27_agent_alive_check alive, void *opaque,
                                 q27_agent_tool_result *result) {
    int parent = -1;
    char *leaf = NULL;
    struct stat st;
    int fd = open_regular(rootfd, request->path, O_RDONLY,
                          &parent, &leaf, &st, result);
    if (fd < 0) return Q27_AGENT_OK;
    close(parent);
    free(leaf);
    if ((uint64_t)st.st_size > request->max_output_bytes) {
        result_error(result, "read output exceeds configured bound");
        close(fd);
        return Q27_AGENT_OK;
    }
    unsigned char buffer[16384];
    size_t remaining = (size_t)st.st_size;
    while (remaining) {
        if (!alive(opaque)) { close(fd); return Q27_AGENT_CANCELLED; }
        size_t want = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        ssize_t n = read(fd, buffer, want);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0) errno = EIO;
            result_errno(result, "cannot read tool file");
            close(fd);
            return Q27_AGENT_OK;
        }
        if (!sink(buffer, (size_t)n, opaque)) {
            close(fd);
            return Q27_AGENT_CANCELLED;
        }
        result->output_bytes += (uint32_t)n;
        remaining -= (size_t)n;
    }
    close(fd);
    result->exit_code = 0;
    return Q27_AGENT_OK;
}

static q27_agent_status run_search(int rootfd,
                                   const q27_agent_tool_request *request,
                                   q27_agent_tool_sink sink,
                                   q27_agent_alive_check alive, void *opaque,
                                   q27_agent_tool_result *result) {
    int parent = -1;
    char *leaf = NULL;
    struct stat st;
    int fd = open_regular(rootfd, request->path, O_RDONLY,
                          &parent, &leaf, &st, result);
    if (fd < 0) return Q27_AGENT_OK;
    close(parent);
    free(leaf);
    unsigned char *data = read_all(fd, (size_t)st.st_size, result);
    close(fd);
    if (!data) return Q27_AGENT_OK;
    int cancelled = 0;
    size_t *table = kmp_table(request->input, request->input_len,
                              alive, opaque, &cancelled);
    if (!table) {
        free(data);
        if (cancelled) return Q27_AGENT_CANCELLED;
        result_error(result, "out of memory building search matcher");
        return Q27_AGENT_OK;
    }

    unsigned char *out = malloc(request->max_output_bytes ?
                                request->max_output_bytes : 1);
    if (!out) {
        free(table);
        free(data);
        result_error(result, "out of memory");
        return Q27_AGENT_OK;
    }
    size_t used = 0, line_start = 0;
    uint64_t line_number = 1;
    while (line_start < (size_t)st.st_size) {
        if (!alive(opaque)) {
            free(out); free(table); free(data);
            return Q27_AGENT_CANCELLED;
        }
        size_t line_end = line_start;
        while (line_end < (size_t)st.st_size && data[line_end] != '\n') ++line_end;
        size_t line_len = line_end - line_start;
        if (kmp_contains(data + line_start, line_len,
                         request->input, request->input_len, table,
                         alive, opaque, &cancelled)) {
            char prefix[32];
            int prefix_len = snprintf(prefix, sizeof(prefix), "%llu:",
                                      (unsigned long long)line_number);
            size_t newline = line_end < (size_t)st.st_size ? 1 : 0;
            if (prefix_len < 0 || (size_t)prefix_len > request->max_output_bytes - used ||
                line_len > request->max_output_bytes - used - (size_t)prefix_len ||
                newline > request->max_output_bytes - used - (size_t)prefix_len - line_len) {
                result_error(result, "search output exceeds configured bound");
                free(out);
                free(table);
                free(data);
                return Q27_AGENT_OK;
            }
            memcpy(out + used, prefix, (size_t)prefix_len);
            used += (size_t)prefix_len;
            memcpy(out + used, data + line_start, line_len + newline);
            used += line_len + newline;
        }
        if (cancelled) {
            free(out); free(table); free(data);
            return Q27_AGENT_CANCELLED;
        }
        if (line_end == (size_t)st.st_size) break;
        line_start = line_end + 1;
        ++line_number;
    }
    free(table);
    free(data);
    if (used && !sink(out, used, opaque)) {
        free(out);
        return Q27_AGENT_CANCELLED;
    }
    free(out);
    result->output_bytes = (uint32_t)used;
    result->exit_code = 0;
    return Q27_AGENT_OK;
}

static int write_all(int fd, const unsigned char *data, size_t len) {
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = write(fd, data + offset, len - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        offset += (size_t)n;
    }
    return 1;
}

static uint64_t monotonic_ms(void);

static int atomic_swap_at(int parent, const char *a, const char *b) {
#if defined(__APPLE__)
    return renameatx_np(parent, a, parent, b, RENAME_SWAP);
#elif defined(__linux__) && defined(SYS_renameat2)
    return (int)syscall(SYS_renameat2, parent, a, parent, b,
                        RENAME_EXCHANGE);
#else
    errno = ENOTSUP;
    return -1;
#endif
}

static int atomic_create_between(int source_parent, const char *source,
                                 int target_parent, const char *target) {
#if defined(__APPLE__)
    return renameatx_np(source_parent, source, target_parent, target,
                        RENAME_EXCL);
#elif defined(__linux__) && defined(SYS_renameat2)
    return (int)syscall(SYS_renameat2, source_parent, source,
                        target_parent, target, RENAME_NOREPLACE);
#else
    errno = ENOTSUP;
    return -1;
#endif
}

static int remove_pinned_regular_file(int parent, const char *name,
                                      int file_fd) {
    struct stat pinned, named;
    if (fstat(file_fd, &pinned) != 0 ||
        fstatat(parent, name, &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        pinned.st_dev != named.st_dev || pinned.st_ino != named.st_ino)
        return 0;
    return unlinkat(parent, name, 0) == 0;
}

static int remove_pinned_empty_directory(int parent, const char *name,
                                         int directory_fd) {
    struct stat pinned, named;
    if (fstat(directory_fd, &pinned) != 0 ||
        fstatat(parent, name, &named, AT_SYMLINK_NOFOLLOW) != 0 ||
        pinned.st_dev != named.st_dev || pinned.st_ino != named.st_ino)
        return 0;
    return unlinkat(parent, name, AT_REMOVEDIR) == 0;
}

static q27_agent_status run_write(int rootfd,
                                  const q27_agent_tool_request *request,
                                  q27_agent_alive_check alive, void *opaque,
                                  q27_agent_tool_result *result) {
    int parent = -1;
    char *leaf = NULL;
    if (!open_parent(rootfd, request->path, &parent, &leaf, result))
        return Q27_AGENT_OK;
    struct stat existing;
    if (fstatat(parent, leaf, &existing, AT_SYMLINK_NOFOLLOW) == 0) {
        result_error(result, "write target already exists; use edit");
        close(parent);
        free(leaf);
        return Q27_AGENT_OK;
    }
    if (errno != ENOENT) {
        result_errno(result, "cannot inspect write target");
        close(parent);
        free(leaf);
        return Q27_AGENT_OK;
    }
    if (!alive(opaque)) {
        close(parent);
        free(leaf);
        return Q27_AGENT_CANCELLED;
    }
    // Atomic create cannot safely clean up pathname races in a directory
    // writable by another principal. Require the pinned destination parent to
    // be owned by this account and non-writable by group/other. Same-account
    // processes are already inside the harness's authority boundary.
    struct stat parent_stat;
    if (fstat(parent, &parent_stat) != 0 || !S_ISDIR(parent_stat.st_mode) ||
        parent_stat.st_uid != geteuid() || (parent_stat.st_mode & 0022)) {
        result_error(result,
                     "write parent must be owner-owned and not group/other writable");
        close(parent);
        free(leaf);
        return Q27_AGENT_OK;
    }
    int parent_acl = fd_has_granting_acl(parent);
    if (parent_acl != 0) {
        result_error(result, parent_acl > 0 ?
                     "write parent has a granting extended ACL" :
                     "cannot inspect write parent ACL");
        close(parent);
        free(leaf);
        return Q27_AGENT_OK;
    }

    // Keep the payload beneath a pinned owner-only directory. A principal
    // that can rename entries in the destination parent still cannot replace
    // the source entry used by renameatx_np/renameat2.
    char stage_name[96];
    int stage_fd = -1;
    struct stat created_stage;
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        snprintf(stage_name, sizeof(stage_name), ".q27-write-stage-%ld-%u",
                 (long)getpid(), attempt);
        if (mkdirat(parent, stage_name, 0700) == 0) {
            // mkdir mode is umask-filtered. Inspect before chmod and later
            // require the opened fd to retain this exact identity.
            if (fstatat(parent, stage_name, &created_stage,
                        AT_SYMLINK_NOFOLLOW) == 0 &&
                S_ISDIR(created_stage.st_mode) &&
                created_stage.st_uid == geteuid() &&
                fchmodat(parent, stage_name, 0700,
                         AT_SYMLINK_NOFOLLOW) == 0)
                stage_fd = openat(parent, stage_name,
                                  O_RDONLY | O_DIRECTORY | O_NOFOLLOW |
                                  O_CLOEXEC);
            // Without a pinned fd, pathname cleanup could delete a
            // concurrently substituted directory. Preserve the empty residue
            // on failure rather than acting on an unverified name.
            break;
        }
        if (errno != EEXIST) break;
    }
    if (stage_fd < 0) {
        result_errno(result, "cannot create private write staging directory");
        close(parent);
        free(leaf);
        return Q27_AGENT_OK;
    }
    struct stat stage = {0}, named_stage = {0};
    int stage_path_owned =
        fchmod(stage_fd, 0700) == 0 && clear_extended_acl_fd(stage_fd) &&
        fstat(stage_fd, &stage) == 0 &&
        stage.st_uid == geteuid() && S_ISDIR(stage.st_mode) &&
        stage.st_dev == created_stage.st_dev &&
        stage.st_ino == created_stage.st_ino &&
        (stage.st_mode & 07777) == 0700 &&
        fstatat(parent, stage_name, &named_stage, AT_SYMLINK_NOFOLLOW) == 0 &&
        named_stage.st_dev == stage.st_dev && named_stage.st_ino == stage.st_ino;
    if (!stage_path_owned) {
        result_error(result, "write staging directory identity changed");
        // ACL/mode verification can fail while the pathname still names our
        // pinned, empty directory. Remove only that exact inode; on mismatch
        // preserve the foreign entry rather than deleting by name.
        struct stat cleanup_fd, cleanup_name;
        int cleanup_owned =
            fstat(stage_fd, &cleanup_fd) == 0 &&
            fstatat(parent, stage_name, &cleanup_name,
                    AT_SYMLINK_NOFOLLOW) == 0 &&
            cleanup_fd.st_dev == cleanup_name.st_dev &&
            cleanup_fd.st_ino == cleanup_name.st_ino &&
            cleanup_fd.st_dev == created_stage.st_dev &&
            cleanup_fd.st_ino == created_stage.st_ino;
        close(stage_fd);
        if (cleanup_owned) unlinkat(parent, stage_name, AT_REMOVEDIR);
        close(parent);
        free(leaf);
        return Q27_AGENT_OK;
    }

    static const char payload_name[] = "payload";
    int temp_fd = openat(stage_fd, payload_name,
                         O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                         0600);
    if (temp_fd < 0) {
        result_errno(result, "cannot create private write payload");
        remove_pinned_empty_directory(parent, stage_name, stage_fd);
        close(stage_fd);
        close(parent);
        free(leaf);
        return Q27_AGENT_OK;
    }

    int cancelled = 0, prepared = 1, renamed = 0, published = 0;
    struct stat staged;
    size_t offset = 0;
    while (offset < request->input_len) {
        if (!alive(opaque)) { cancelled = 1; prepared = 0; break; }
        ssize_t n = write(temp_fd, request->input + offset,
                          request->input_len - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { prepared = 0; break; }
        offset += (size_t)n;
    }
    if (prepared && !alive(opaque)) {
        cancelled = 1;
        prepared = 0;
    }
    // Creation mode is filtered through umask; force the exact private mode.
    if (prepared && (fchmod(temp_fd, 0600) != 0 ||
                     !clear_extended_acl_fd(temp_fd) || fsync(temp_fd) != 0 ||
                     fstat(temp_fd, &staged) != 0))
        prepared = 0;
    if (prepared) {
        unsigned char *verified = read_all(temp_fd, request->input_len, result);
        const int content_ok = verified &&
            (!request->input_len ||
             !memcmp(verified, request->input, request->input_len));
        free(verified);
        if (!content_ok) { errno = EIO; prepared = 0; }
    }
    if (prepared && !alive(opaque)) {
        cancelled = 1;
        prepared = 0;
    }
    if (prepared) {
        struct stat named;
        if (fstatat(stage_fd, payload_name, &named, AT_SYMLINK_NOFOLLOW) != 0 ||
            !S_ISREG(named.st_mode) || named.st_dev != staged.st_dev ||
            named.st_ino != staged.st_ino || named.st_nlink != 1 ||
            named.st_size != (off_t)request->input_len) {
            errno = EBUSY;
            prepared = 0;
        }
    }
    if (prepared) {
        if (atomic_create_between(stage_fd, payload_name, parent, leaf) == 0) {
            renamed = 1;
            const int dirsync_result = fsync(parent);
            const int dirsync_errno = errno;
            struct stat installed;
            const int identity_ok =
                fstatat(parent, leaf, &installed, AT_SYMLINK_NOFOLLOW) == 0 &&
                S_ISREG(installed.st_mode) && installed.st_dev == staged.st_dev &&
                installed.st_ino == staged.st_ino && installed.st_nlink == 1 &&
                installed.st_size == (off_t)request->input_len &&
                (installed.st_mode & 07777) == 0600;
            if (identity_ok) {
                published = 1;
                result->exit_code = 0;
                if (dirsync_result != 0) {
                    result->flags |= Q27_TOOL_FLAG_DURABILITY_UNCERTAIN;
                    snprintf(result->message, sizeof(result->message),
                             "write committed; directory fsync failed: %s",
                             strerror(dirsync_errno));
                }
            } else {
                // Do not unlink or quarantine by pathname here: after an
                // identity mismatch the visible entry may belong to a
                // concurrent creator. The failed transaction reports the
                // uncertain side effect without deleting foreign content.
                result->flags |= Q27_TOOL_FLAG_DURABILITY_UNCERTAIN;
                result_error(result,
                             "atomic write publication failed validation");
            }
        } else if (errno == EEXIST) {
            result_error(result, "write target already exists; use edit");
        }
    }
    if (!published && result->exit_code == -1 && !result->message[0] &&
        !cancelled)
        result_errno(result, "cannot publish atomic write");

    const int saved = errno;
    if (!renamed)
        remove_pinned_regular_file(stage_fd, payload_name, temp_fd);
    close(temp_fd);
    // Only remove the parent entry when it still names our pinned directory;
    // a concurrent rename may leave an empty private residue but cannot expose
    // payload bytes or redirect publication.
    remove_pinned_empty_directory(parent, stage_name, stage_fd);
    close(stage_fd);
    close(parent);
    free(leaf);
    errno = saved;
    return cancelled ? Q27_AGENT_CANCELLED : Q27_AGENT_OK;
}

static q27_agent_status run_edit(int rootfd,
                                 const q27_agent_tool_request *request,
                                 q27_agent_alive_check alive, void *opaque,
                                 q27_agent_tool_result *result) {
    int parent = -1;
    char *leaf = NULL;
    struct stat before;
    int fd = open_regular(rootfd, request->path, O_RDWR,
                          &parent, &leaf, &before, result);
    if (fd < 0) return Q27_AGENT_OK;
    const uint64_t lock_deadline = monotonic_ms() + 5000;
    for (;;) {
        // Lock the stable parent directory inode, not the destination file:
        // atomic exchange replaces the destination inode mid-transaction.
        // This serializes all q27 edits in one directory through validation,
        // cleanup, and any rollback.
        if (flock(parent, LOCK_EX | LOCK_NB) == 0) break;
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
            result_errno(result, "cannot lock edit target");
            close(fd); close(parent); free(leaf);
            return Q27_AGENT_OK;
        }
        if (!alive(opaque)) {
            close(fd); close(parent); free(leaf);
            return Q27_AGENT_CANCELLED;
        }
        if (monotonic_ms() >= lock_deadline) {
            result_error(result, "edit lock timed out");
            close(fd); close(parent); free(leaf);
            return Q27_AGENT_OK;
        }
        struct timespec nap = {.tv_sec = 0, .tv_nsec = 10 * 1000 * 1000};
        nanosleep(&nap, NULL);
    }
    if (!alive(opaque)) {
        close(fd); close(parent); free(leaf);
        return Q27_AGENT_CANCELLED;
    }
    if (fstat(fd, &before) != 0) {
        result_errno(result, "cannot restat locked edit target");
        close(fd); close(parent); free(leaf);
        return Q27_AGENT_OK;
    }
    if (!S_ISREG(before.st_mode) || before.st_size < 0 ||
        (uint64_t)before.st_size > FILE_MAX_BYTES) {
        errno = EINVAL;
        result_errno(result, "edit target changed while locking");
        close(fd); close(parent); free(leaf);
        return Q27_AGENT_OK;
    }
    unsigned char *data = read_all(fd, (size_t)before.st_size, result);
    if (!data) { close(fd); close(parent); free(leaf); return Q27_AGENT_OK; }
    int cancelled = 0;
    size_t *table = kmp_table(request->input, request->input_len,
                              alive, opaque, &cancelled);
    if (!table) {
        free(data); close(fd); close(parent); free(leaf);
        if (cancelled) return Q27_AGENT_CANCELLED;
        result_error(result, "out of memory building edit matcher");
        return Q27_AGENT_OK;
    }
    size_t matches = 0, match_at = 0;
    if (!kmp_matches(data, (size_t)before.st_size,
                     request->input, request->input_len, table,
                     alive, opaque, &matches, &match_at, &cancelled)) {
        free(table); free(data); close(fd); close(parent); free(leaf);
        return Q27_AGENT_CANCELLED;
    }
    free(table);
    if (matches != 1) {
        result_error(result, matches ? "edit old bytes are not unique" :
                                       "edit old bytes were not found");
        free(data); close(fd); close(parent); free(leaf);
        return Q27_AGENT_OK;
    }
    size_t old_size = (size_t)before.st_size;
    if (request->replacement_len > SIZE_MAX - (old_size - request->input_len) ||
        old_size - request->input_len + request->replacement_len > FILE_MAX_BYTES) {
        result_error(result, "edited file exceeds bound");
        free(data); close(fd); close(parent); free(leaf);
        return Q27_AGENT_OK;
    }
    size_t new_size = old_size - request->input_len + request->replacement_len;
    unsigned char *updated = malloc(new_size ? new_size : 1);
    if (!updated) {
        result_error(result, "out of memory");
        free(data); close(fd); close(parent); free(leaf);
        return Q27_AGENT_OK;
    }
    memcpy(updated, data, match_at);
    if (request->replacement_len)
        memcpy(updated + match_at, request->replacement, request->replacement_len);
    memcpy(updated + match_at + request->replacement_len,
           data + match_at + request->input_len,
           old_size - match_at - request->input_len);
    if (!alive(opaque)) {
        free(updated); free(data); close(fd); close(parent); free(leaf);
        return Q27_AGENT_CANCELLED;
    }

    char temp[96];
    int temp_fd = -1;
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        snprintf(temp, sizeof(temp), ".q27-edit-%ld-%u.tmp",
                 (long)getpid(), attempt);
        temp_fd = openat(parent, temp, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
                         before.st_mode & 07777);
        if (temp_fd >= 0 || errno != EEXIST) break;
    }
    if (temp_fd < 0) {
        result_errno(result, "cannot create edit temporary file");
        free(updated); free(data); close(fd); close(parent); free(leaf);
        return Q27_AGENT_OK;
    }
    int published = 0, cancelled_before_swap = 0, preserve_temp = 0;
    struct stat prepared;
    if (write_all(temp_fd, updated, new_size) &&
        fchmod(temp_fd, before.st_mode & 07777) == 0 && fsync(temp_fd) == 0 &&
        fstat(temp_fd, &prepared) == 0) {
        if (!alive(opaque)) {
            cancelled_before_swap = 1;
        } else {
            // Atomic exchange captures the exact destination at the temp
            // name. Validate that displaced inode and its bytes against the
            // locked snapshot; on any mismatch, atomically swap back.
            if (atomic_swap_at(parent, temp, leaf) == 0) {
                struct stat displaced, installed;
                unsigned char *verify_old = NULL, *verify_new = NULL;
                int identity_ok =
                    fstatat(parent, temp, &displaced, AT_SYMLINK_NOFOLLOW) == 0 &&
                    displaced.st_dev == before.st_dev &&
                    displaced.st_ino == before.st_ino &&
                    displaced.st_size == before.st_size &&
                    fstatat(parent, leaf, &installed, AT_SYMLINK_NOFOLLOW) == 0 &&
                    installed.st_dev == prepared.st_dev &&
                    installed.st_ino == prepared.st_ino &&
                    installed.st_size == (off_t)new_size;
                if (identity_ok) {
                    verify_old = read_all(fd, old_size, result);
                    verify_new = read_all(temp_fd, new_size, result);
                }
                int content_ok = identity_ok && verify_old && verify_new &&
                                 !memcmp(verify_old, data, old_size) &&
                                 !memcmp(verify_new, updated, new_size);
                free(verify_old);
                free(verify_new);
                if (content_ok) {
                    if (unlinkat(parent, temp, 0) == 0) {
                        // The visible atomic exchange is now committed and
                        // the recovery inode is gone. A directory-fsync
                        // failure cannot be rolled back truthfully: report
                        // success plus explicit durability uncertainty.
                        published = 1;
                        if (fsync(parent) != 0) {
                            result->flags |=
                                Q27_TOOL_FLAG_DURABILITY_UNCERTAIN;
                            snprintf(result->message, sizeof(result->message),
                                     "edit committed; directory fsync failed: %s",
                                     strerror(errno));
                        }
                    } else {
                        int rollback_ok =
                            atomic_swap_at(parent, temp, leaf) == 0;
                        if (rollback_ok) {
                            unlinkat(parent, temp, 0);
                            (void)fsync(parent);
                        } else {
                            preserve_temp = 1;
                        }
                    }
                } else {
                    int rollback_ok = atomic_swap_at(parent, temp, leaf) == 0;
                    if (rollback_ok) {
                        unlinkat(parent, temp, 0);
                        (void)fsync(parent);
                    } else {
                        // The displaced original remains recoverable at temp;
                        // never unlink it after a failed rollback.
                        preserve_temp = 1;
                    }
                    errno = EBUSY;
                }
            }
        }
    }
    if (temp_fd >= 0) close(temp_fd);
    if (!published) {
        int saved = errno;
        if (!preserve_temp) unlinkat(parent, temp, 0);
        errno = saved ? saved : EIO;
        if (!cancelled_before_swap)
            result_errno(result, "cannot publish atomic edit");
    } else {
        result->exit_code = 0;
    }
    free(updated);
    free(data);
    close(fd);
    close(parent);
    free(leaf);
    return cancelled_before_swap ? Q27_AGENT_CANCELLED : Q27_AGENT_OK;
}

static uint64_t monotonic_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000 + (uint64_t)now.tv_nsec / 1000000;
}

static void terminate_group(pid_t pid) {
    if (pid <= 0) return;
    kill(-pid, SIGTERM);
    const uint64_t deadline = monotonic_ms() + 200;
    int status;
    while (monotonic_ms() < deadline) {
        pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid || (waited < 0 && errno == ECHILD)) {
            // The shell leader may exit before background descendants.
            kill(-pid, SIGKILL);
            return;
        }
        struct timespec nap = {.tv_sec = 0, .tv_nsec = 10 * 1000 * 1000};
        nanosleep(&nap, NULL);
    }
    kill(-pid, SIGKILL);
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
}

#if defined(__linux__)
static int install_no_fork_filter(void) {
#if defined(__x86_64__)
#define Q27_AUDIT_ARCH AUDIT_ARCH_X86_64
#elif defined(__aarch64__)
#define Q27_AUDIT_ARCH AUDIT_ARCH_AARCH64
#else
    errno = ENOTSUP;
    return 0;
#endif
    struct sock_filter code[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (unsigned int)offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, Q27_AUDIT_ARCH, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 (unsigned int)offsetof(struct seccomp_data, nr)),
#if defined(__x86_64__) && defined(__X32_SYSCALL_BIT)
        // x32 shares AUDIT_ARCH_X86_64 but ORs a compatibility bit into
        // syscall numbers; reject that ABI rather than allow unknown clones.
        BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, __X32_SYSCALL_BIT, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
#endif
#ifdef __NR_clone
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
#endif
#ifdef __NR_clone3
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_clone3, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
#endif
#ifdef __NR_fork
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_fork, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
#endif
#ifdef __NR_vfork
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_vfork, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EPERM),
#endif
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
    };
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(code) / sizeof(code[0])),
        .filter = code};
#undef Q27_AUDIT_ARCH
    return prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0 &&
           prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0;
}
#endif

static q27_agent_status run_shell(int workspace_fd,
                                  const q27_agent_tool_request *request,
                                  q27_agent_tool_sink sink,
                                  q27_agent_alive_check alive, void *opaque,
                                  q27_agent_tool_result *result) {
    char *command = malloc(request->input_len + 1);
    if (!command) { result_error(result, "out of memory"); return Q27_AGENT_OK; }
    memcpy(command, request->input, request->input_len);
    command[request->input_len] = '\0';
    int pipes[2];
    if (pipe(pipes) != 0) {
        result_errno(result, "cannot create shell output pipe");
        free(command);
        return Q27_AGENT_OK;
    }
    fcntl(pipes[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipes[1], F_SETFD, FD_CLOEXEC);
    pid_t pid = fork();
    if (pid == 0) {
        setpgid(0, 0);
        close(pipes[0]);
        if (dup2(pipes[1], STDOUT_FILENO) < 0 ||
            dup2(pipes[1], STDERR_FILENO) < 0 ||
            fchdir(workspace_fd) != 0)
            _exit(126);
        close(pipes[1]);
#if defined(__APPLE__)
        // A no-fork sandbox makes the job a genuinely bounded process tree:
        // shell builtins and a final exec work; pipelines/background jobs fail.
        execl("/usr/bin/sandbox-exec", "sandbox-exec", "-p",
              "(version 1)(allow default)(deny process-fork)",
              "/bin/sh", "-c", command, (char *)NULL);
#elif defined(__linux__)
        if (!install_no_fork_filter()) _exit(125);
        execl("/bin/sh", "sh", "-c", command, (char *)NULL);
#else
        _exit(125);
#endif
        _exit(127);
    }
    free(command);
    close(pipes[1]);
    if (pid < 0) {
        result_errno(result, "cannot start shell job");
        close(pipes[0]);
        return Q27_AGENT_OK;
    }
    setpgid(pid, pid);
    result->exit_code = 0; // running; result_error switches this to -1
    int flags = fcntl(pipes[0], F_GETFL);
    if (flags >= 0) fcntl(pipes[0], F_SETFL, flags | O_NONBLOCK);

    const uint64_t deadline = monotonic_ms() + request->timeout_ms;
    int status = 0, child_done = 0, pipe_done = 0;
    uint64_t post_leader_deadline = 0;
    q27_agent_status call_status = Q27_AGENT_OK;
    unsigned char buffer[16384];
    while (!pipe_done || !child_done) {
        if (!alive(opaque)) {
            call_status = Q27_AGENT_CANCELLED;
            break;
        }
        const uint64_t now = monotonic_ms();
        if (child_done && !pipe_done && post_leader_deadline &&
            now >= post_leader_deadline) {
            // A daemonized process may have escaped the shell's group while
            // retaining the pipe. The leader result remains authoritative;
            // stop draining after the bounded post-leader grace period.
            pipe_done = 1;
            continue;
        }
        if (now >= deadline) {
            result->flags |= Q27_TOOL_FLAG_TIMED_OUT;
            result_error(result, "shell job timed out");
            break;
        }
        for (;;) {
            ssize_t n = read(pipes[0], buffer, sizeof(buffer));
            if (n > 0) {
                if ((uint64_t)result->output_bytes + (uint64_t)n >
                    request->max_output_bytes) {
                    result->flags |= Q27_TOOL_FLAG_OUTPUT_LIMIT;
                    result_error(result, "shell output exceeds configured bound");
                    pipe_done = 1;
                    break;
                }
                if (!sink(buffer, (size_t)n, opaque)) {
                    call_status = Q27_AGENT_CANCELLED;
                    pipe_done = 1;
                    break;
                }
                result->output_bytes += (uint32_t)n;
                continue;
            }
            if (n == 0) pipe_done = 1;
            else if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                result_errno(result, "cannot read shell output");
                pipe_done = 1;
            }
            break;
        }
        if (call_status == Q27_AGENT_CANCELLED || result->exit_code == -1)
            break;
        pid_t waited = waitpid(pid, &status, WNOHANG);
        if (waited == pid) {
            child_done = 1;
            // Retire background members as soon as the authoritative shell
            // leader exits, then drain any already-buffered output briefly.
            terminate_group(pid);
            post_leader_deadline = monotonic_ms() + 250;
        }
        else if (waited < 0 && errno != EINTR) {
            result_errno(result, "cannot wait for shell job");
            break;
        }
        if (!pipe_done) {
            struct pollfd pfd = {.fd = pipes[0], .events = POLLIN};
            (void)poll(&pfd, 1, 25);
        } else if (!child_done) {
            // Polling an EOF/HUP pipe returns immediately; sleep while only
            // the child terminal remains outstanding.
            struct timespec nap = {
                .tv_sec = 0, .tv_nsec = 25 * 1000 * 1000};
            nanosleep(&nap, NULL);
        }
    }
    close(pipes[0]);
    if (call_status == Q27_AGENT_CANCELLED || result->exit_code == -1 ||
        !child_done)
        terminate_group(pid);
    else {
        if (WIFEXITED(status)) result->exit_code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) {
            result->flags |= Q27_TOOL_FLAG_SIGNALED;
            result->exit_code = 128 + WTERMSIG(status);
        } else result_error(result, "shell job ended without terminal status");
    }
    return call_status;
}

q27_agent_status q27_agent_tool_execute(
    int workspace_fd, const q27_agent_tool_request *request,
    q27_agent_tool_sink sink, q27_agent_alive_check alive, void *opaque,
    q27_agent_tool_result *result) {
    if (!result) return Q27_AGENT_REJECTED;
    memset(result, 0, sizeof(*result));
    result->exit_code = -1;
    if (workspace_fd < 0 || !sink || !alive) {
        result_error(result, "invalid workspace or callbacks");
        return Q27_AGENT_OK;
    }
    if (!request_valid(request, result)) return Q27_AGENT_OK;
    if (!alive(opaque)) return Q27_AGENT_CANCELLED;
    if (request->kind == Q27_TOOL_SHELL)
        return run_shell(workspace_fd, request, sink, alive, opaque, result);

    int rootfd = dup(workspace_fd);
    if (rootfd < 0) {
        result_errno(result, "cannot duplicate workspace root");
        return Q27_AGENT_OK;
    }
    q27_agent_status status;
    if (request->kind == Q27_TOOL_READ)
        status = run_read(rootfd, request, sink, alive, opaque, result);
    else if (request->kind == Q27_TOOL_SEARCH)
        status = run_search(rootfd, request, sink, alive, opaque, result);
    else if (request->kind == Q27_TOOL_WRITE)
        status = run_write(rootfd, request, alive, opaque, result);
    else
        status = run_edit(rootfd, request, alive, opaque, result);
    close(rootfd);
    return status;
}
