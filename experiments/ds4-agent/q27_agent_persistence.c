#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "q27_agent_persistence.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define SESSION_MAX_BYTES (64u * 1024u * 1024u)
#define SESSION_MAX_MESSAGES 65536u
#define SESSION_HEADER_BYTES 84u
static const unsigned char session_magic[8] = {'Q','2','7','A','G','T','2','\0'};

typedef struct { unsigned char *p; size_t len; size_t cap; } bytes;

static int read_private_file(const char *path, unsigned char **data, size_t *len,
                             char *error, size_t error_cap);
static int read_private_fd(int fd, unsigned char **data, size_t *len,
                           char *error, size_t error_cap);
static int manifest_snapshot_name_at(int dfd, const char *base, char **name,
                                     char *error, size_t error_cap);

static void set_error(char *out, size_t cap, const char *text) {
    if (out && cap) snprintf(out, cap, "%s", text ? text : "session error");
}

static void set_errno_error(char *out, size_t cap, const char *what) {
    if (out && cap) snprintf(out, cap, "%s: %s", what, strerror(errno));
}

static int append_bytes(bytes *b, const void *p, size_t n) {
    if (n > SESSION_MAX_BYTES || b->len > SESSION_MAX_BYTES - n) return 0;
    const size_t need = b->len + n;
    if (need > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < need) {
            if (cap > SESSION_MAX_BYTES / 2) { cap = SESSION_MAX_BYTES; break; }
            cap *= 2;
        }
        unsigned char *grown = realloc(b->p, cap);
        if (!grown) return 0;
        b->p = grown;
        b->cap = cap;
    }
    if (n) memcpy(b->p + b->len, p, n);
    b->len += n;
    return 1;
}

static int put_u32(bytes *b, uint32_t v) {
    unsigned char p[4] = {(unsigned char)v, (unsigned char)(v >> 8),
                          (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
    return append_bytes(b, p, sizeof(p));
}

static int put_u64(bytes *b, uint64_t v) {
    unsigned char p[8];
    for (unsigned i = 0; i < 8; ++i) p[i] = (unsigned char)(v >> (8 * i));
    return append_bytes(b, p, sizeof(p));
}

static uint32_t get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t get_u64(const unsigned char *p) {
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

static uint32_t crc32_bytes(const unsigned char *p, size_t n) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}

static int split_path(const char *path, char **dir, char **base,
                      char *error, size_t error_cap) {
    *dir = *base = NULL;
    if (!path || !*path || path[strlen(path) - 1] == '/') {
        set_error(error, error_cap, "session path must name a file");
        return 0;
    }
    const char *slash = strrchr(path, '/');
    const char *name = slash ? slash + 1 : path;
    if (!*name || !strcmp(name, ".") || !strcmp(name, "..")) {
        set_error(error, error_cap, "invalid session filename");
        return 0;
    }
    if (slash) {
        size_t n = (size_t)(slash - path);
        *dir = n ? strndup(path, n) : strdup("/");
    } else {
        *dir = strdup(".");
    }
    *base = strdup(name);
    if (!*dir || !*base) {
        free(*dir); free(*base); *dir = *base = NULL;
        set_error(error, error_cap, "out of memory splitting session path");
        return 0;
    }
    return 1;
}

static int valid_snapshot_name(const char *name) {
    if (!name || !*name || strlen(name) > 255 ||
        !strcmp(name, ".") || !strcmp(name, "..") || strchr(name, '/')) return 0;
    return strstr(name, ".q27snap.") != NULL;
}

static int snapshot_name_matches_manifest(const char *name,
                                          const char *manifest_base) {
    if (!valid_snapshot_name(name) || !manifest_base) return 0;
    const size_t base_len = strlen(manifest_base);
    const char suffix[] = ".q27snap.";
    if (strlen(name) < 1 + base_len + sizeof(suffix) - 1) return 0;
    return name[0] == '.' && !memcmp(name + 1, manifest_base, base_len) &&
           !memcmp(name + 1 + base_len, suffix, sizeof(suffix) - 1);
}

static char *join_path(const char *dir, const char *name) {
    size_t a = strlen(dir), b = strlen(name);
    if (a > SIZE_MAX - b - 2) return NULL;
    char *out = malloc(a + b + 2);
    if (!out) return NULL;
    memcpy(out, dir, a);
    size_t at = a;
    if (!a || dir[a - 1] != '/') out[at++] = '/';
    memcpy(out + at, name, b + 1);
    return out;
}

static int random_hex(char out[33]) {
    unsigned char raw[16];
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    size_t at = 0;
    while (at < sizeof(raw)) {
        ssize_t n = read(fd, raw + at, sizeof(raw) - at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); errno = EIO; return 0; }
        at += (size_t)n;
    }
    close(fd);
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < sizeof(raw); ++i) {
        out[2*i] = hex[raw[i] >> 4]; out[2*i+1] = hex[raw[i] & 15];
    }
    out[32] = '\0';
    return 1;
}

static int protected_directory(int dfd, char *error, size_t error_cap) {
    struct stat st;
    if (fstat(dfd, &st) != 0 || !S_ISDIR(st.st_mode) ||
        st.st_uid != geteuid() || (st.st_mode & 0022) != 0) {
        set_error(error, error_cap,
                  "session directory must be owner-controlled and not group/other writable");
        errno = EPERM;
        return 0;
    }
    return 1;
}

static int private_regular_at(int dfd, const char *name, int require_exists,
                              char *error, size_t error_cap) {
    struct stat st;
    if (fstatat(dfd, name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
        if (!require_exists && errno == ENOENT) return 1;
        set_errno_error(error, error_cap, "cannot inspect session file");
        return 0;
    }
    if (!S_ISREG(st.st_mode) || st.st_uid != geteuid() ||
        (st.st_mode & 0777) != 0600) {
        set_error(error, error_cap,
                  "session files must be owner-only regular files (mode 0600)");
        errno = EPERM;
        return 0;
    }
    return 1;
}

int q27_agent_session_new_snapshot_path(const char *manifest_path,
                                        char **snapshot_path,
                                        char **snapshot_name,
                                        char *error, size_t error_cap) {
    if (snapshot_path) *snapshot_path = NULL;
    if (snapshot_name) *snapshot_name = NULL;
    if (!snapshot_path || !snapshot_name) {
        set_error(error, error_cap, "invalid snapshot path output");
        return 0;
    }
    char *dir = NULL, *base = NULL;
    if (!split_path(manifest_path, &dir, &base, error, error_cap)) return 0;
    // Resolve the caller's explicit parent path once (macOS /tmp is itself a
    // symlink), then pin all publication operations to this descriptor.
    int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0) {
        set_errno_error(error, error_cap, "cannot open session directory");
        free(dir); free(base); return 0;
    }
    if (!protected_directory(dfd, error, error_cap)) {
        close(dfd); free(dir); free(base); return 0;
    }
    int ok = 0;
    for (unsigned attempt = 0; attempt < 32 && !ok; ++attempt) {
        char random[33];
        if (!random_hex(random)) break;
        size_t need = strlen(base) + sizeof("..q27snap.") + strlen(random);
        char *name = malloc(need);
        if (!name) { errno = ENOMEM; break; }
        snprintf(name, need, ".%s.q27snap.%s", base, random);
        struct stat st;
        if (fstatat(dfd, name, &st, AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT) {
            char *path = join_path(dir, name);
            if (path) { *snapshot_name = name; *snapshot_path = path; ok = 1; }
            else { free(name); errno = ENOMEM; }
        } else free(name);
    }
    if (!ok) set_errno_error(error, error_cap, "cannot allocate snapshot filename");
    close(dfd); free(dir); free(base);
    return ok;
}

static int role_code(const char *role, size_t index) {
    if (!role) return 0;
    if (index == 0) return !strcmp(role, "system") ? 1 : 0;
    if (index & 1) return !strcmp(role, "user") ? 2 : 0;
    return !strcmp(role, "assistant") ? 3 : 0;
}

static int encode_manifest(bytes *out, const char *snapshot_name,
                           const q27_agent_message *messages, size_t message_count,
                           int thinking, int tools, uint32_t context,
                           const unsigned char tokenizer_sha1[20],
                           const unsigned char snapshot_sha256[32],
                           char *error, size_t error_cap) {
    if (!valid_snapshot_name(snapshot_name) || !messages || !tokenizer_sha1 ||
        !snapshot_sha256 || message_count < 3 || message_count > SESSION_MAX_MESSAGES ||
        !(message_count & 1) || context < 2) {
        set_error(error, error_cap, "invalid session manifest fields");
        return 0;
    }
    bytes body = {0};
    if (!append_bytes(&body, snapshot_name, strlen(snapshot_name))) goto oom;
    for (size_t i = 0; i < message_count; ++i) {
        int code = role_code(messages[i].role, i);
        if (!code || !messages[i].content || messages[i].content_len > SESSION_MAX_BYTES) {
            free(body.p); set_error(error, error_cap,
                "session transcript must be system then complete user/assistant pairs");
            return 0;
        }
        unsigned char c = (unsigned char)code;
        if (!append_bytes(&body, &c, 1) ||
            !put_u64(&body, messages[i].content_len) ||
            !append_bytes(&body, messages[i].content, messages[i].content_len)) goto oom;
    }
    if (!append_bytes(out, session_magic, sizeof(session_magic)) ||
        !put_u32(out, (thinking ? 1u : 0u) | (tools ? 2u : 0u)) ||
        !put_u32(out, context) || !put_u32(out, (uint32_t)message_count) ||
        !put_u32(out, (uint32_t)strlen(snapshot_name)) ||
        !put_u64(out, body.len) || !append_bytes(out, tokenizer_sha1, 20) ||
        !append_bytes(out, snapshot_sha256, 32) ||
        !append_bytes(out, body.p, body.len)) goto oom;
    free(body.p);
    return put_u32(out, crc32_bytes(out->p, out->len));
oom:
    free(body.p); set_error(error, error_cap, "session manifest exceeds 64 MiB");
    return 0;
}

static int write_all(int fd, const unsigned char *p, size_t n) {
    while (n) {
        ssize_t wrote = write(fd, p, n);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote <= 0) return 0;
        p += wrote; n -= (size_t)wrote;
    }
    return 1;
}

int q27_agent_session_publish(const char *manifest_path,
                              const char *snapshot_path,
                              const char *snapshot_name,
                              const char *old_snapshot_name,
                              const q27_agent_message *messages,
                              size_t message_count,
                              int enable_thinking,
                              int enable_tools,
                              uint32_t context,
                              const unsigned char tokenizer_sha1[20],
                              const unsigned char snapshot_sha256[32],
                              char *error, size_t error_cap) {
    char *dir = NULL, *base = NULL;
    if (!split_path(manifest_path, &dir, &base, error, error_cap)) return 0;
    char *expected = snapshot_name_matches_manifest(snapshot_name, base) ?
                     join_path(dir, snapshot_name) : NULL;
    if (!expected || !snapshot_path || strcmp(expected, snapshot_path)) {
        set_error(error, error_cap, "snapshot must be in the session directory");
        free(expected); free(dir); free(base); return 0;
    }
    free(expected);
    bytes encoded = {0};
    if (!encode_manifest(&encoded, snapshot_name, messages, message_count,
                         enable_thinking, enable_tools, context,
                         tokenizer_sha1, snapshot_sha256,
                         error, error_cap)) {
        free(dir); free(base); free(encoded.p); return 0;
    }
    int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0) {
        set_errno_error(error, error_cap, "cannot open session directory");
        free(encoded.p); free(dir); free(base); return 0;
    }
    if (!protected_directory(dfd, error, error_cap)) {
        close(dfd); free(encoded.p); free(dir); free(base); return 0;
    }
    size_t lock_len = strlen(base) + sizeof("..lock");
    char *lock_name = malloc(lock_len);
    if (!lock_name) {
        set_error(error, error_cap, "out of memory");
        close(dfd); free(encoded.p); free(dir); free(base); return 0;
    }
    snprintf(lock_name, lock_len, ".%s.lock", base);
    int lock_fd = openat(dfd, lock_name,
                         O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (lock_fd < 0 || fchmod(lock_fd, 0600) != 0 ||
        flock(lock_fd, LOCK_EX) != 0) {
        set_errno_error(error, error_cap, "cannot lock session manifest");
        if (lock_fd >= 0) close(lock_fd);
        free(lock_name); close(dfd); free(encoded.p); free(dir); free(base);
        return 0;
    }
    free(lock_name);

    // The lock serializes publication; this comparison makes it optimistic
    // concurrency control as well. A writer that generated from an older
    // manifest fails instead of silently replacing a newer completed turn.
    int current_matches = 0;
    if (old_snapshot_name) {
        char *current_name = NULL;
        current_matches = manifest_snapshot_name_at(
            dfd, base, &current_name, error, error_cap) &&
            !strcmp(current_name, old_snapshot_name);
        free(current_name);
    } else {
        struct stat current;
        current_matches = fstatat(dfd, base, &current,
                                  AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT;
    }
    if (!current_matches) {
        set_error(error, error_cap,
                  "session manifest changed since this writer loaded it");
        close(lock_fd); close(dfd); free(encoded.p); free(dir); free(base);
        return 0;
    }
    if (!private_regular_at(dfd, snapshot_name, 1, error, error_cap) ||
        !private_regular_at(dfd, base, 0, error, error_cap)) {
        close(lock_fd); close(dfd); free(encoded.p); free(dir); free(base);
        return 0;
    }
    char random[33];
    if (!random_hex(random)) {
        set_errno_error(error, error_cap, "cannot create manifest temporary");
        close(lock_fd); close(dfd); free(encoded.p); free(dir); free(base);
        return 0;
    }
    size_t temp_len = strlen(base) + strlen(random) + 7;
    char *temp = malloc(temp_len);
    if (!temp) {
        set_error(error, error_cap, "out of memory");
        close(lock_fd); close(dfd); free(encoded.p); free(dir); free(base);
        return 0;
    }
    snprintf(temp, temp_len, ".%s.%s.tmp", base, random);
    int fd = openat(dfd, temp, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    int ok = fd >= 0 && fchmod(fd, 0600) == 0 &&
             write_all(fd, encoded.p, encoded.len) && fsync(fd) == 0;
    if (fd >= 0 && close(fd) != 0) ok = 0;
    int result = 0;
    if (ok && renameat(dfd, temp, dfd, base) != 0) ok = 0;
    if (!ok) {
        int saved = errno ? errno : EIO;
        unlinkat(dfd, temp, 0);
        errno = saved;
        set_errno_error(error, error_cap, "cannot publish session manifest");
    } else {
        int sync_result = fsync(dfd);
#ifdef Q27_AGENT_PERSISTENCE_TESTING
        if (getenv("Q27_AGENT_TEST_DIR_FSYNC_FAIL")) {
            errno = EIO;
            sync_result = -1;
        }
#endif
        if (sync_result != 0) {
            // The namespace may already expose the new manifest. Report
            // uncertainty separately so its referenced snapshot is retained.
            set_errno_error(error, error_cap,
                            "session manifest renamed but directory sync failed");
            result = 2;
        } else {
            result = 1;
            if (valid_snapshot_name(old_snapshot_name) &&
                strcmp(old_snapshot_name, snapshot_name)) {
                // Cleanup is post-commit and best effort. A crash can leave an
                // orphan, but never a committed manifest with a missing blob.
                (void)unlinkat(dfd, old_snapshot_name, 0);
                (void)fsync(dfd);
            }
        }
    }
    free(temp); close(lock_fd); close(dfd);
    free(encoded.p); free(dir); free(base);
    return result;
}

int q27_agent_session_discard(const char *manifest_path,
                              const char *expected_snapshot_name,
                              char *error, size_t error_cap) {
    char *dir = NULL, *base = NULL;
    if (!manifest_path ||
        !split_path(manifest_path, &dir, &base, error, error_cap))
        return 0;
    int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0) {
        set_errno_error(error, error_cap, "cannot open session directory");
        free(dir); free(base); return 0;
    }
    if (!protected_directory(dfd, error, error_cap)) {
        close(dfd); free(dir); free(base); return 0;
    }
    size_t lock_len = strlen(base) + sizeof("..lock");
    char *lock_name = malloc(lock_len);
    if (!lock_name) {
        set_error(error, error_cap, "out of memory");
        close(dfd); free(dir); free(base); return 0;
    }
    snprintf(lock_name, lock_len, ".%s.lock", base);
    int lock_fd = openat(dfd, lock_name,
                         O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (lock_fd < 0 || fchmod(lock_fd, 0600) != 0 ||
        flock(lock_fd, LOCK_EX) != 0) {
        set_errno_error(error, error_cap, "cannot lock session for discard");
        if (lock_fd >= 0) close(lock_fd);
        free(lock_name); close(dfd); free(dir); free(base);
        return 0;
    }
    free(lock_name);

    struct stat current;
    int missing = fstatat(dfd, base, &current, AT_SYMLINK_NOFOLLOW) != 0 &&
                  errno == ENOENT;
    if (missing) {
        /* Manifest already gone — still require durable snapshot cleanup. */
        int cleaned = 0;
        if (expected_snapshot_name &&
            valid_snapshot_name(expected_snapshot_name)) {
            if (unlinkat(dfd, expected_snapshot_name, 0) != 0 &&
                errno != ENOENT) {
                set_errno_error(error, error_cap,
                                "cannot unlink expected session snapshot");
                close(lock_fd); close(dfd); free(dir); free(base);
                return 0;
            }
            cleaned = 1;
        }
        if (cleaned) {
            int sync_result = fsync(dfd);
#ifdef Q27_AGENT_PERSISTENCE_TESTING
            if (getenv("Q27_AGENT_TEST_DIR_FSYNC_FAIL")) {
                errno = EIO;
                sync_result = -1;
            }
#endif
            if (sync_result != 0) {
                set_errno_error(error, error_cap,
                                "session discarded but directory sync failed");
                close(lock_fd); close(dfd); free(dir); free(base);
                return 2;
            }
        }
        close(lock_fd); close(dfd); free(dir); free(base);
        return 1;
    }

    char *live_name = NULL;
    /* CAS against the pinned directory, not the original pathname (symlink
     * races on a parent path must not delete a different directory's files). */
    if (!manifest_snapshot_name_at(dfd, base, &live_name, error, error_cap)) {
        close(lock_fd); close(dfd); free(dir); free(base);
        return 0;
    }
    if (!expected_snapshot_name) {
        /* No in-process CAS baseline: refuse to delete a live session that
         * another process may have published after we started empty. */
        set_error(error, error_cap,
                  "session exists but this process has no expected snapshot");
        free(live_name); close(lock_fd); close(dfd); free(dir); free(base);
        return 0;
    }
    if (strcmp(live_name, expected_snapshot_name) != 0) {
        set_error(error, error_cap,
                  "session manifest changed since this process loaded it");
        free(live_name); close(lock_fd); close(dfd); free(dir); free(base);
        return 0;
    }

    if (unlinkat(dfd, base, 0) != 0 && errno != ENOENT) {
        set_errno_error(error, error_cap, "cannot unlink session manifest");
        free(live_name); close(lock_fd); close(dfd); free(dir); free(base);
        return 0;
    }
    /* Manifest is gone from this process's view. Later failures must report
     * partial success (2) so callers drop CAS expectations. */
    if (valid_snapshot_name(live_name)) {
        if (unlinkat(dfd, live_name, 0) != 0 && errno != ENOENT) {
            set_errno_error(error, error_cap, "cannot unlink session snapshot");
            free(live_name); close(lock_fd); close(dfd); free(dir); free(base);
            return 2;
        }
    }
    int sync_result = fsync(dfd);
#ifdef Q27_AGENT_PERSISTENCE_TESTING
    if (getenv("Q27_AGENT_TEST_DIR_FSYNC_FAIL")) {
        errno = EIO;
        sync_result = -1;
    }
#endif
    if (sync_result != 0) {
        set_errno_error(error, error_cap,
                        "session discarded but directory sync failed");
        free(live_name); close(lock_fd); close(dfd); free(dir); free(base);
        return 2;
    }
    free(live_name); close(lock_fd); close(dfd); free(dir); free(base);
    return 1;
}

static int read_private_fd(int fd, unsigned char **data, size_t *len,
                           char *error, size_t error_cap) {
    *data = NULL; *len = 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != geteuid() ||
        (st.st_mode & 077) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size > SESSION_MAX_BYTES) {
        errno = EPERM;
        set_error(error, error_cap, "session manifest is not a private bounded regular file");
        return 0;
    }
    size_t n = (size_t)st.st_size;
    unsigned char *p = malloc(n ? n : 1);
    if (!p) { set_error(error, error_cap, "out of memory"); return 0; }
    size_t at = 0;
    while (at < n) {
        ssize_t got = read(fd, p + at, n - at);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { free(p); set_error(error, error_cap, "truncated session manifest"); return 0; }
        at += (size_t)got;
    }
    unsigned char extra;
    ssize_t more;
    do { more = read(fd, &extra, 1); } while (more < 0 && errno == EINTR);
    if (more != 0) { free(p); set_error(error, error_cap, "session manifest changed while reading"); return 0; }
    *data = p; *len = n; return 1;
}

static int read_private_file(const char *path, unsigned char **data, size_t *len,
                             char *error, size_t error_cap) {
    *data = NULL; *len = 0;
    int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) { set_errno_error(error, error_cap, "cannot open session manifest"); return 0; }
    int ok = read_private_fd(fd, data, len, error, error_cap);
    close(fd);
    return ok;
}

static int parse_manifest_snapshot_name(const unsigned char *data, size_t len,
                                        char **name, char *error,
                                        size_t error_cap) {
    *name = NULL;
    int ok = len >= SESSION_HEADER_BYTES + 4 &&
             !memcmp(data, session_magic, 8) &&
             crc32_bytes(data, len - 4) == get_u32(data + len - 4);
    uint32_t snap_len = ok ? get_u32(data + 20) : 0;
    uint64_t body_len = ok ? get_u64(data + 24) : 0;
    if (!ok || !snap_len || snap_len > 255 || snap_len > body_len ||
        body_len != len - SESSION_HEADER_BYTES - 4 ||
        memchr(data + SESSION_HEADER_BYTES, '\0', snap_len)) {
        set_error(error, error_cap, "invalid current Q27AGT2 manifest");
        return 0;
    }
    *name = strndup((const char *)data + SESSION_HEADER_BYTES, snap_len);
    if (!*name || !valid_snapshot_name(*name)) {
        free(*name); *name = NULL;
        set_error(error, error_cap, "invalid current snapshot name");
        return 0;
    }
    return 1;
}

static int manifest_snapshot_name_at(int dfd, const char *base, char **name,
                                     char *error, size_t error_cap) {
    *name = NULL;
    int fd = openat(dfd, base, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        set_errno_error(error, error_cap, "cannot open session manifest");
        return 0;
    }
    unsigned char *data = NULL;
    size_t len = 0;
    int ok = read_private_fd(fd, &data, &len, error, error_cap);
    close(fd);
    if (!ok) return 0;
    ok = parse_manifest_snapshot_name(data, len, name, error, error_cap);
    free(data);
    return ok;
}

static int message_contains(const q27_agent_message *message,
                            const char *needle, size_t needle_len) {
    if (!message || !message->content || !needle_len ||
        message->content_len < needle_len) return 0;
    const unsigned char *p = (const unsigned char *)message->content;
    for (size_t i = 0; i <= message->content_len - needle_len; ++i)
        if (!memcmp(p + i, needle, needle_len)) return 1;
    return 0;
}

static int is_raw_payload_request(const q27_agent_message *message) {
    static const char write_prefix[] =
        "<q27_raw_payload_request version=\"2\" kind=\"write\">\n";
    static const char edit_prefix[] =
        "<q27_raw_payload_request version=\"2\" kind=\"edit\">\n";
    static const char suffix[] = "\n</q27_raw_payload_request>";
    if (!message || !message->role || strcmp(message->role, "user") ||
        !message->content) return 0;
    const size_t prefix_len =
        message->content_len >= sizeof(write_prefix) - 1 &&
        !memcmp(message->content, write_prefix, sizeof(write_prefix) - 1) ?
        sizeof(write_prefix) - 1 :
        message->content_len >= sizeof(edit_prefix) - 1 &&
        !memcmp(message->content, edit_prefix, sizeof(edit_prefix) - 1) ?
        sizeof(edit_prefix) - 1 : 0;
    return prefix_len && message->content_len >= prefix_len + sizeof(suffix) - 1 &&
        !memcmp(message->content + message->content_len - (sizeof(suffix) - 1),
                suffix, sizeof(suffix) - 1);
}

// Automatic tool traffic is indivisible for compaction. Ordinary calls are
// assistant-call/user-response pairs. Bulk write/edit calls additionally have
// a harness-authored user request and assistant raw payload between them.
static int is_automatic_tool_user(const q27_agent_message *messages, size_t i) {
    static const char response[] = "<tool_response>\n";
    static const char call_open[] = "<tool_call>";
    static const char call_close[] = "</tool_call>";
    if (!messages || i < 2 || !messages[i].role ||
        strcmp(messages[i].role, "user"))
        return 0;
    if (is_raw_payload_request(&messages[i]))
        return messages[i-1].role &&
            !strcmp(messages[i-1].role, "assistant") &&
            message_contains(&messages[i-1], call_open, sizeof(call_open) - 1) &&
            message_contains(&messages[i-1], call_close, sizeof(call_close) - 1);
    if (!messages[i].content ||
        messages[i].content_len < sizeof(response) - 1 ||
        memcmp(messages[i].content, response, sizeof(response) - 1))
        return 0;
    if (messages[i-1].role && !strcmp(messages[i-1].role, "assistant") &&
        message_contains(&messages[i-1], call_open, sizeof(call_open) - 1) &&
        message_contains(&messages[i-1], call_close, sizeof(call_close) - 1))
        return 1;
    return i >= 4 && messages[i-1].role &&
        !strcmp(messages[i-1].role, "assistant") &&
        is_raw_payload_request(&messages[i-2]) &&
        messages[i-3].role && !strcmp(messages[i-3].role, "assistant") &&
        message_contains(&messages[i-3], call_open, sizeof(call_open) - 1) &&
        message_contains(&messages[i-3], call_close, sizeof(call_close) - 1);
}

int q27_agent_compaction_cut(const q27_agent_message *messages,
                             size_t message_count,
                             uint32_t keep_root_turns,
                             size_t *cut_index) {
    if (cut_index) *cut_index = 0;
    if (!messages || !cut_index || !keep_root_turns || message_count < 5 ||
        strcmp(messages[0].role ? messages[0].role : "", "system"))
        return 0;
    size_t roots = 0;
    for (size_t i = 1; i < message_count; ++i)
        if (messages[i].role && !strcmp(messages[i].role, "user") &&
            !is_automatic_tool_user(messages, i))
            ++roots;
    if (roots <= keep_root_turns) return 0;
    const size_t wanted = roots - keep_root_turns;
    roots = 0;
    for (size_t i = 1; i < message_count; ++i) {
        if (messages[i].role && !strcmp(messages[i].role, "user") &&
            !is_automatic_tool_user(messages, i) && roots++ == wanted) {
            *cut_index = i;
            return i > 1;
        }
    }
    return 0;
}

static int session_reader_lock(const char *manifest_path, int *lock_fd,
                               char *error, size_t error_cap) {
    *lock_fd = -1;
    char *dir = NULL, *base = NULL;
    if (!split_path(manifest_path, &dir, &base, error, error_cap)) return 0;
    int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd < 0 || !protected_directory(dfd, error, error_cap)) {
        if (dfd >= 0) close(dfd);
        free(dir); free(base);
        return 0;
    }
    size_t lock_len = strlen(base) + sizeof("..lock");
    char *lock_name = malloc(lock_len);
    if (!lock_name) {
        close(dfd); free(dir); free(base);
        set_error(error, error_cap, "out of memory");
        return 0;
    }
    snprintf(lock_name, lock_len, ".%s.lock", base);
    int fd = openat(dfd, lock_name,
                    O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0600);
    free(lock_name); close(dfd); free(dir); free(base);
    if (fd < 0 || fchmod(fd, 0600) != 0 || flock(fd, LOCK_SH) != 0) {
        if (fd >= 0) close(fd);
        set_errno_error(error, error_cap, "cannot lock session for load");
        return 0;
    }
    *lock_fd = fd;
    return 1;
}

void q27_agent_saved_session_free(q27_agent_saved_session *session) {
    if (!session) return;
    for (size_t i = 0; i < session->message_count; ++i) {
        free((char *)session->messages[i].role);
        free((char *)session->messages[i].content);
    }
    free(session->messages); free(session->snapshot_path); free(session->snapshot_name);
    if (session->lock_held) close(session->lock_fd);
    *session = (q27_agent_saved_session){0};
}

int q27_agent_session_load(const char *manifest_path,
                           q27_agent_saved_session *session,
                           char *error, size_t error_cap) {
    if (!session) { set_error(error, error_cap, "invalid session output"); return 0; }
    *session = (q27_agent_saved_session){0};
    int reader_lock = -1;
    if (!session_reader_lock(manifest_path, &reader_lock, error, error_cap))
        return 0;
    unsigned char *data = NULL; size_t len = 0;
    if (!read_private_file(manifest_path, &data, &len, error, error_cap)) {
        close(reader_lock);
        return 0;
    }
    if (len < SESSION_HEADER_BYTES + 4 || memcmp(data, session_magic, 8) ||
        crc32_bytes(data, len - 4) != get_u32(data + len - 4)) {
        free(data); close(reader_lock);
        set_error(error, error_cap, "invalid or corrupt Q27AGT2 manifest"); return 0;
    }
    uint32_t flags = get_u32(data + 8), context = get_u32(data + 12);
    uint32_t count = get_u32(data + 16), snap_len = get_u32(data + 20);
    uint64_t body_len = get_u64(data + 24);
    if ((flags & ~3u) || context < 2 || count < 3 || count > SESSION_MAX_MESSAGES ||
        !(count & 1) || !snap_len || snap_len > 255 ||
        body_len != len - SESSION_HEADER_BYTES - 4 || snap_len > body_len) {
        free(data); close(reader_lock);
        set_error(error, error_cap, "invalid Q27AGT2 header"); return 0;
    }
    const unsigned char *snap_bytes = data + SESSION_HEADER_BYTES;
    char *snap = memchr(snap_bytes, '\0', snap_len) ? NULL :
                 strndup((const char *)snap_bytes, snap_len);
    if (!snap || !valid_snapshot_name(snap)) {
        free(snap); free(data); close(reader_lock);
        set_error(error, error_cap, "invalid snapshot name in manifest"); return 0;
    }
    q27_agent_message *messages = calloc(count, sizeof(*messages));
    if (!messages) {
        free(snap); free(data); close(reader_lock);
        set_error(error, error_cap, "out of memory"); return 0;
    }
    size_t at = SESSION_HEADER_BYTES + snap_len;
    int ok = 1;
    for (uint32_t i = 0; i < count && ok; ++i) {
        if (at > len - 4 || len - 4 - at < 9) { ok = 0; break; }
        unsigned code = data[at++]; uint64_t n64 = get_u64(data + at); at += 8;
        const char *role = i == 0 ? "system" : (i & 1) ? "user" : "assistant";
        unsigned expected = i == 0 ? 1 : (i & 1) ? 2 : 3;
        if (code != expected || n64 > SESSION_MAX_BYTES || n64 > len - 4 - at) { ok = 0; break; }
        char *r = strdup(role), *content = malloc((size_t)n64 + 1);
        if (!r || !content) { free(r); free(content); ok = 0; break; }
        memcpy(content, data + at, (size_t)n64); content[n64] = '\0'; at += (size_t)n64;
        messages[i] = (q27_agent_message){.role=r,.content=content,.content_len=(size_t)n64};
    }
    if (at != len - 4) ok = 0;
    char *dir = NULL, *base = NULL, *snapshot_path = NULL;
    if (ok && split_path(manifest_path, &dir, &base, error, error_cap) &&
        snapshot_name_matches_manifest(snap, base))
        snapshot_path = join_path(dir, snap);
    else {
        ok = 0;
        if (error && error_cap)
            set_error(error, error_cap,
                      "snapshot name does not belong to this manifest");
    }
    if (ok && !snapshot_path) ok = 0;
    if (ok) {
        int dfd = open(dir, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dfd < 0 || !protected_directory(dfd, error, error_cap) ||
            !private_regular_at(dfd, snap, 1, error, error_cap)) ok = 0;
        if (dfd >= 0) close(dfd);
    }
    unsigned char tokenizer_sha1[20], snapshot_sha256[32];
    memcpy(tokenizer_sha1, data + 32, sizeof(tokenizer_sha1));
    memcpy(snapshot_sha256, data + 52, sizeof(snapshot_sha256));
    free(dir); free(base); free(data);
    if (!ok) {
        q27_agent_saved_session partial = {.messages=messages,.message_count=count,
            .snapshot_path=snapshot_path,.snapshot_name=snap,
            .lock_fd=reader_lock,.lock_held=1};
        q27_agent_saved_session_free(&partial);
        if (!error || !error[0]) set_error(error, error_cap, "invalid Q27AGT2 body");
        return 0;
    }
    session->messages = messages; session->message_count = count;
    session->enable_thinking = !!(flags & 1); session->enable_tools = !!(flags & 2);
    session->context = context;
    memcpy(session->tokenizer_sha1, tokenizer_sha1, 20);
    memcpy(session->snapshot_sha256, snapshot_sha256, 32);
    session->snapshot_path = snapshot_path; session->snapshot_name = snap;
    session->lock_fd = reader_lock;
    session->lock_held = 1;
    return 1;
}
