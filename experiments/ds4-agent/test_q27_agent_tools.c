#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "q27_agent_tools.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    unsigned char *data;
    size_t len;
    size_t cap;
    int alive;
    unsigned alive_calls;
    unsigned cancel_after;
    const char *replace_staging_path;
    int replaced_staging;
} capture;

static int alive(void *opaque) {
    capture *state = opaque;
    ++state->alive_calls;
    if (state->replace_staging_path && !state->replaced_staging &&
        access(state->replace_staging_path, F_OK) == 0) {
        if (unlink(state->replace_staging_path) == 0) {
            int fd = open(state->replace_staging_path,
                          O_WRONLY | O_CREAT | O_EXCL, 0600);
            if (fd >= 0) {
                ssize_t written = write(fd, "attacker", 8);
                int closed = close(fd);
                state->replaced_staging = written == 8 && closed == 0;
            }
        }
    }
    if (state->cancel_after && state->alive_calls >= state->cancel_after)
        state->alive = 0;
    return state->alive;
}

static int sink(const unsigned char *bytes, size_t len, void *opaque) {
    capture *out = opaque;
    if (len > SIZE_MAX - out->len) return 0;
    if (out->len + len > out->cap) {
        size_t cap = out->cap ? out->cap : 64;
        while (cap < out->len + len) cap *= 2;
        unsigned char *grown = realloc(out->data, cap);
        if (!grown) return 0;
        out->data = grown;
        out->cap = cap;
    }
    memcpy(out->data + out->len, bytes, len);
    out->len += len;
    return 1;
}

static void capture_reset(capture *out) {
    free(out->data);
    *out = (capture){.alive = 1};
}

static int write_file(const char *path, const void *data, size_t len, mode_t mode) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, mode);
    if (fd < 0) return 0;
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, (const unsigned char *)data + off, len - off);
        if (n <= 0) { close(fd); return 0; }
        off += (size_t)n;
    }
    return close(fd) == 0;
}

#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL: %s\n", message); return 1; } \
} while (0)

int main(void) {
    char root[] = "/tmp/q27-agent-tools-XXXXXX";
    CHECK(mkdtemp(root), "temporary workspace created");
    int workspace_fd = open(root, O_RDONLY | O_DIRECTORY);
    CHECK(workspace_fd >= 0, "workspace fd pinned");
    char sub[512], file[512], duplicate[512], linkpath[512], root_file[512];
    char created[512], cancelled_write[512], empty_file[512];
    char raced_write[512], raced_stage_dir[512], raced_staging[512];
    char acl_parent[512], shared_parent[512];
    snprintf(sub, sizeof(sub), "%s/sub", root);
    snprintf(file, sizeof(file), "%s/sub/data.bin", root);
    snprintf(duplicate, sizeof(duplicate), "%s/sub/duplicate.txt", root);
    snprintf(linkpath, sizeof(linkpath), "%s/sub/link", root);
    snprintf(root_file, sizeof(root_file), "%s/root.txt", root);
    snprintf(created, sizeof(created), "%s/sub/created.bin", root);
    snprintf(cancelled_write, sizeof(cancelled_write), "%s/sub/cancelled.bin", root);
    snprintf(empty_file, sizeof(empty_file), "%s/sub/empty.bin", root);
    snprintf(raced_write, sizeof(raced_write), "%s/sub/raced.bin", root);
    snprintf(raced_stage_dir, sizeof(raced_stage_dir),
             "%s/sub/.q27-write-stage-%ld-0", root, (long)getpid());
    snprintf(raced_staging, sizeof(raced_staging), "%s/payload",
             raced_stage_dir);
    snprintf(acl_parent, sizeof(acl_parent), "%s/acl-parent", root);
    snprintf(shared_parent, sizeof(shared_parent), "%s/shared-parent", root);
    CHECK(mkdir(sub, 0700) == 0, "workspace subdirectory created");
    const unsigned char original[] = {'a','l','p','h','a','\n','x','\0','n','e','e','d','l','e','\n'};
    CHECK(write_file(file, original, sizeof(original), 0640), "binary fixture written");
    CHECK(write_file(duplicate, "old old", 7, 0600), "duplicate fixture written");
    CHECK(write_file(root_file, "first", 5, 0600), "root edit fixture written");
    CHECK(symlink("data.bin", linkpath) == 0, "symlink fixture created");

    capture out = {.alive = 1};
    q27_agent_tool_result result;
    q27_agent_tool_request request = {
        .kind = Q27_TOOL_READ, .path = "sub/data.bin",
        .timeout_ms = 1000, .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == 0 &&
          out.len == sizeof(original) && !memcmp(out.data, original, sizeof(original)),
          "read preserves exact binary bytes");

    char moved[512], replacement_sub[512], replacement_file[512];
    snprintf(moved, sizeof(moved), "%s.moved", root);
    snprintf(replacement_sub, sizeof(replacement_sub), "%s/sub", root);
    snprintf(replacement_file, sizeof(replacement_file), "%s/sub/data.bin", root);
    CHECK(rename(root, moved) == 0 && mkdir(root, 0700) == 0 &&
          mkdir(replacement_sub, 0700) == 0 &&
          write_file(replacement_file, "outside", 7, 0600),
          "workspace pathname replacement fixture created");
    capture_reset(&out);
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == 0 &&
          out.len == sizeof(original) && !memcmp(out.data, original, sizeof(original)),
          "pinned workspace fd ignores pathname replacement");
    unlink(replacement_file);
    rmdir(replacement_sub);
    rmdir(root);
    CHECK(rename(moved, root) == 0, "workspace pathname restored");

    capture_reset(&out);
    const unsigned char needle[] = {'\0','n','e','e'};
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_SEARCH, .path = "sub/data.bin",
        .input = needle, .input_len = sizeof(needle),
        .max_output_bytes = 1024};
    const unsigned char expected_search[] = {'2',':','x','\0','n','e','e','d','l','e','\n'};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == 0 &&
          out.len == sizeof(expected_search) &&
          !memcmp(out.data, expected_search, sizeof(expected_search)),
          "search returns numbered binary-safe matching line");

    const unsigned char new_content[] = {'n','e','w','\0','f','i','l','e'};
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_WRITE, .path = "sub/created.bin",
        .input = new_content, .input_len = sizeof(new_content),
        .max_output_bytes = 1024};
    q27_agent_tool_request write_preflight = request;
    write_preflight.kind = Q27_TOOL_WRITE_PREFLIGHT;
    write_preflight.input = NULL;
    write_preflight.input_len = 0;
    CHECK(q27_agent_tool_execute(workspace_fd, &write_preflight, sink, alive,
                                 &out, &result) == Q27_AGENT_OK &&
          result.exit_code == 0 && access(created, F_OK) != 0,
          "write preflight validates without creating the destination");
    mode_t saved_umask = umask(0777);
    q27_agent_status write_status = q27_agent_tool_execute(
        workspace_fd, &request, sink, alive, &out, &result);
    umask(saved_umask);
    CHECK(write_status == Q27_AGENT_OK &&
          result.exit_code == 0 && result.output_bytes == 0,
          "write atomically creates a binary file under restrictive umask");
    CHECK(q27_agent_tool_execute(workspace_fd, &write_preflight, sink, alive,
                                 &out, &result) == Q27_AGENT_OK &&
          result.exit_code == -1 && strstr(result.message, "already exists"),
          "write preflight rejects an existing destination");
    struct stat created_st;
    unsigned char created_bytes[sizeof(new_content)];
    int created_fd = open(created, O_RDONLY);
    CHECK(created_fd >= 0 && fstat(created_fd, &created_st) == 0 &&
          (created_st.st_mode & 0777) == 0600 &&
          read(created_fd, created_bytes, sizeof(created_bytes)) ==
              (ssize_t)sizeof(created_bytes) &&
          !memcmp(created_bytes, new_content, sizeof(new_content)),
          "write publishes exact bytes with fixed owner-only mode");
    close(created_fd);
    const unsigned char overwrite[] = "overwrite";
    request.input = overwrite;
    request.input_len = sizeof(overwrite) - 1;
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          result.exit_code == -1 && strstr(result.message, "already exists"),
          "write refuses to overwrite an existing file");
    created_fd = open(created, O_RDONLY);
    CHECK(created_fd >= 0 &&
          read(created_fd, created_bytes, sizeof(created_bytes)) ==
              (ssize_t)sizeof(created_bytes) &&
          !memcmp(created_bytes, new_content, sizeof(new_content)),
          "failed overwrite leaves existing bytes intact");
    close(created_fd);

    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_WRITE, .path = "sub/empty.bin",
        .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          result.exit_code == 0 && stat(empty_file, &created_st) == 0 &&
          created_st.st_size == 0,
          "write creates an empty file");

    out.cancel_after = 2;
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_WRITE, .path = "sub/cancelled.bin",
        .input = new_content, .input_len = sizeof(new_content),
        .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_CANCELLED &&
          access(cancelled_write, F_OK) != 0,
          "cancelled write never publishes a partial destination");
    out.alive = 1;
    out.alive_calls = out.cancel_after = 0;

    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_WRITE, .path = "sub/raced.bin",
        .input = new_content, .input_len = sizeof(new_content),
        .max_output_bytes = 1024};
    out.replace_staging_path = raced_staging;
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          out.replaced_staging && result.exit_code == -1 &&
          access(raced_write, F_OK) != 0 &&
          access(raced_staging, F_OK) == 0,
          "replaced staging pathname is rejected without deleting foreign content");
    CHECK(unlink(raced_staging) == 0 && rmdir(raced_stage_dir) == 0,
          "foreign staging fixture removed by its creator");

    capture_reset(&out);
    const unsigned char old[] = {'x','\0','n','e','e','d','l','e'};
    const unsigned char replacement[] = {'r','e','p','l','a','c','e','d'};
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_EDIT, .path = "sub/data.bin",
        .input = old, .input_len = sizeof(old),
        .replacement = replacement, .replacement_len = sizeof(replacement),
        .max_output_bytes = 1024};
    q27_agent_tool_request edit_preflight = request;
    edit_preflight.kind = Q27_TOOL_EDIT_PREFLIGHT;
    edit_preflight.replacement = NULL;
    edit_preflight.replacement_len = 0;
    CHECK(q27_agent_tool_execute(workspace_fd, &edit_preflight, sink, alive,
                                 &out, &result) == Q27_AGENT_OK &&
          result.exit_code == 0,
          "edit preflight validates a unique old-byte match without mutation");
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == 0,
          "unique binary edit publishes");
    struct stat st;
    CHECK(stat(file, &st) == 0 && (st.st_mode & 0777) == 0640,
          "atomic edit preserves mode");
    unsigned char edited[15];
    int fd = open(file, O_RDONLY);
    CHECK(fd >= 0 && read(fd, edited, sizeof(edited)) == (ssize_t)sizeof(edited),
          "edited file readable");
    close(fd);
    CHECK(!memcmp(edited, "alpha\nreplaced\n", sizeof(edited)),
          "atomic edit replaced exact bytes");

    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_EDIT, .path = "root.txt",
        .input = (const unsigned char *)"first", .input_len = 5,
        .replacement = (const unsigned char *)"second", .replacement_len = 6,
        .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          result.exit_code == 0,
          "first root-level edit succeeds");
    request.input = (const unsigned char *)"second";
    request.input_len = 6;
    request.replacement = (const unsigned char *)"third";
    request.replacement_len = 5;
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          result.exit_code == 0,
          "second root-level edit proves directory lock release");

    int lock_fd = open(sub, O_RDONLY | O_DIRECTORY);
    CHECK(lock_fd >= 0 && flock(lock_fd, LOCK_EX | LOCK_NB) == 0,
          "edit directory cancellation fixture locked");
    out.cancel_after = 3;
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_EDIT, .path = "sub/duplicate.txt",
        .input = (const unsigned char *)"old", .input_len = 3,
        .replacement = (const unsigned char *)"new", .replacement_len = 3,
        .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_CANCELLED,
          "edit lock wait remains cancellable");
    flock(lock_fd, LOCK_UN);
    close(lock_fd);
    out.alive = 1;
    out.alive_calls = out.cancel_after = 0;

    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_EDIT, .path = "sub/duplicate.txt",
        .input = (const unsigned char *)"old", .input_len = 3,
        .replacement = (const unsigned char *)"new", .replacement_len = 3,
        .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == -1 &&
          strstr(result.message, "not unique"),
          "ambiguous edit fails closed");

    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_READ, .path = "sub/link", .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == -1,
          "final symlink is rejected");
    request.path = "../etc/passwd";
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == -1,
          "parent traversal is rejected");

    CHECK(mkdir(shared_parent, 0700) == 0 && chmod(shared_parent, 0770) == 0,
          "principal-writable parent fixture created");
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_WRITE, .path = "shared-parent/private.txt",
        .input = (const unsigned char *)"private", .input_len = 7,
        .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          result.exit_code == -1 && strstr(result.message, "group/other"),
          "write rejects a parent replaceable by another principal");
    CHECK(rmdir(shared_parent) == 0, "principal-writable fixture removed");

#if defined(__APPLE__)
    CHECK(mkdir(acl_parent, 0700) == 0, "ACL parent fixture created");
    char acl_command[1024];
    snprintf(acl_command, sizeof(acl_command),
             "chmod +a '_www allow list,search,add_file,add_subdirectory,"
             "file_inherit,directory_inherit' %s", acl_parent);
    CHECK(system(acl_command) == 0, "granting inherited ACL fixture installed");
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_WRITE, .path = "acl-parent/private.txt",
        .input = (const unsigned char *)"private", .input_len = 7,
        .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          result.exit_code == -1 && strstr(result.message, "granting"),
          "write rejects a parent with a granting inherited ACL");
    snprintf(acl_command, sizeof(acl_command), "chmod -N %s", acl_parent);
    CHECK(system(acl_command) == 0 && rmdir(acl_parent) == 0,
          "ACL parent fixture removed");
#endif

    capture_reset(&out);
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_SHELL,
        .input = (const unsigned char *)"printf 'a\\000b'",
        .input_len = strlen("printf 'a\\000b'"),
        .timeout_ms = 1000, .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == 0 && out.len == 3 &&
          !memcmp(out.data, "a\0b", 3),
          "shell captures exact binary stdout");

    capture_reset(&out);
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_SHELL,
        .input = (const unsigned char *)"pwd", .input_len = 3,
        .timeout_ms = 1000, .max_output_bytes = 1024};
    char canonical_root[1024];
    CHECK(realpath(root, canonical_root), "workspace canonicalizes");
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == 0 &&
          out.len == strlen(canonical_root) + 1 &&
          !memcmp(out.data, canonical_root, strlen(canonical_root)) &&
          out.data[out.len - 1] == '\n',
          "shell starts inside configured workspace");

    capture_reset(&out);
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_SHELL,
        .input = (const unsigned char *)"sleep 600 &",
        .input_len = strlen("sleep 600 &"),
        .timeout_ms = 1000, .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out,
                                 &result) == Q27_AGENT_OK &&
          result.exit_code != 0,
          "bounded shell rejects background process creation");

    capture_reset(&out);
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_SHELL,
        .input = (const unsigned char *)"sleep 2", .input_len = 7,
        .timeout_ms = 50, .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == -1 &&
          (result.flags & Q27_TOOL_FLAG_TIMED_OUT),
          "shell timeout kills and reaps job");

    capture_reset(&out);
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_SHELL,
        .input = (const unsigned char *)"printf 123456789", .input_len = 16,
        .timeout_ms = 1000, .max_output_bytes = 8};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_OK && result.exit_code == -1 &&
          (result.flags & Q27_TOOL_FLAG_OUTPUT_LIMIT),
          "shell output cap is enforced");

    capture_reset(&out);
    out.alive = 0;
    request = (q27_agent_tool_request){
        .kind = Q27_TOOL_SHELL,
        .input = (const unsigned char *)"sleep 2", .input_len = 7,
        .timeout_ms = 1000, .max_output_bytes = 1024};
    CHECK(q27_agent_tool_execute(workspace_fd, &request, sink, alive, &out, &result) ==
              Q27_AGENT_CANCELLED,
          "pre-launch cancellation is honored");

    capture_reset(&out);
    close(workspace_fd);
    unlink(linkpath);
    unlink(root_file);
    unlink(empty_file);
    unlink(created);
    unlink(duplicate);
    unlink(file);
    rmdir(sub);
    rmdir(root);
    puts("q27 agent tools selftest: PASS");
    return 0;
}
