#define _GNU_SOURCE
#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "q27_agent_persistence.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(c,m) do { if (!(c)) { fprintf(stderr,"FAIL: %s\n",m); return 1; } } while (0)

static int write_private(const char *path, const void *p, size_t n) {
    int fd = open(path, O_WRONLY|O_CREAT|O_TRUNC, 0600);
    if (fd < 0) return 0;
    int ok = write(fd,p,n) == (ssize_t)n && fsync(fd) == 0 && close(fd) == 0;
    if (!ok) close(fd);
    return ok;
}

int main(void) {
    char root[] = "/tmp/q27-agent-persist-XXXXXX";
    CHECK(mkdtemp(root), "temporary directory created");
    char manifest[512]; snprintf(manifest,sizeof(manifest),"%s/chat.q27agent",root);
    char error[256] = {0}; char *snap_path = NULL, *snap_name = NULL;
    char public_dir[512], public_manifest[512];
    snprintf(public_dir,sizeof(public_dir),"%s/public",root);
    snprintf(public_manifest,sizeof(public_manifest),"%s/chat.q27agent",public_dir);
    CHECK(mkdir(public_dir,0700)==0 && chmod(public_dir,0777)==0,
          "unprotected directory fixture created");
    CHECK(!q27_agent_session_new_snapshot_path(public_manifest,&snap_path,
                                                &snap_name,error,sizeof(error)) &&
          strstr(error,"owner-controlled"),
          "group/other writable session directory rejected");
    CHECK(chmod(public_dir,0700)==0 && rmdir(public_dir)==0,
          "unprotected directory fixture removed");
    CHECK(q27_agent_session_new_snapshot_path(manifest,&snap_path,&snap_name,error,sizeof(error)),
          "fresh snapshot name allocated");
    CHECK(strstr(snap_name,".q27snap.") && write_private(snap_path,"Q27SNAP1",8),
          "private snapshot fixture created");
    const char binary[] = {'a','\0','b'};
    unsigned char tokenizer_sha1[20], snapshot_sha256[32];
    for (unsigned i=0;i<20;i++) tokenizer_sha1[i]=(unsigned char)i;
    for (unsigned i=0;i<32;i++) snapshot_sha256[i]=(unsigned char)(31-i);
    q27_agent_message messages[] = {
        {.role="system",.content="system",.content_len=6},
        {.role="user",.content=binary,.content_len=sizeof(binary)},
        {.role="assistant",.content="answer",.content_len=6},
    };
    mode_t prior_umask = umask(0777);
    int first_publish = q27_agent_session_publish(
        manifest,snap_path,snap_name,NULL,messages,3,0,1,8192,
        tokenizer_sha1,snapshot_sha256,error,sizeof(error));
    umask(prior_umask);
    CHECK(first_publish == 1,
          "manifest publishes atomically under a restrictive umask");
    struct stat st; CHECK(stat(manifest,&st)==0 && (st.st_mode&0777)==0600,
                          "manifest is mode 0600");
    q27_agent_saved_session loaded;
    CHECK(q27_agent_session_load(manifest,&loaded,error,sizeof(error)),"manifest loads");
    CHECK(loaded.message_count==3 && !loaded.enable_thinking && loaded.enable_tools &&
          loaded.context==8192 && !memcmp(loaded.tokenizer_sha1,tokenizer_sha1,20) &&
          !memcmp(loaded.snapshot_sha256,snapshot_sha256,32) && !strcmp(loaded.snapshot_name,snap_name) && loaded.messages[1].content_len==3 && !memcmp(loaded.messages[1].content,binary,3),
          "binary transcript and config round trip exactly");
    q27_agent_saved_session_free(&loaded);
    CHECK(q27_agent_session_publish(manifest,snap_path,snap_name,NULL,
                                    messages,3,0,1,8192,tokenizer_sha1,
                                    snapshot_sha256,error,sizeof(error)) == 0 &&
          strstr(error,"changed since"),
          "second fresh writer cannot replace an existing manifest");
    char copied_manifest[512];
    snprintf(copied_manifest,sizeof(copied_manifest),"%s/copied.q27agent",root);
    CHECK(link(manifest,copied_manifest)==0,
          "manifest copy fixture created in same directory");
    CHECK(!q27_agent_session_load(copied_manifest,&loaded,error,sizeof(error)) &&
          strstr(error,"does not belong"),
          "copied manifest cannot share another manifest namespace");
    unlink(copied_manifest);
    char copied_lock[512];
    snprintf(copied_lock,sizeof(copied_lock),"%s/.copied.q27agent.lock",root);
    unlink(copied_lock);

    q27_agent_message tool_history[] = {
        {.role="system",.content="s",.content_len=1},
        {.role="user",.content="task one",.content_len=8},
        {.role="assistant",.content="<tool_call>x</tool_call>",.content_len=24},
        {.role="user",.content="<tool_response>\nresult\n</tool_response>",
         .content_len=sizeof("<tool_response>\nresult\n</tool_response>")-1},
        {.role="assistant",.content="done",.content_len=4},
        {.role="user",.content="task two",.content_len=8},
        {.role="assistant",.content="answer two",.content_len=10},
        {.role="user",.content="task three",.content_len=10},
    };
    size_t cut = 0;
    CHECK(q27_agent_compaction_cut(tool_history, 8, 2, &cut) && cut == 5,
          "compaction retains root turns without splitting tool response pair");
    CHECK(!q27_agent_compaction_cut(tool_history, 8, 3, &cut),
          "compaction defers when no complete old root turn is available");

    q27_agent_message raw_history[] = {
        {.role="system",.content="s",.content_len=1},
        {.role="user",.content="create it",.content_len=9},
        {.role="assistant",.content="<tool_call>x</tool_call>",.content_len=24},
        {.role="user",.content="<q27_raw_payload_request version=\"2\" kind=\"write\">\nx\n</q27_raw_payload_request>",
         .content_len=sizeof("<q27_raw_payload_request version=\"2\" kind=\"write\">\nx\n</q27_raw_payload_request>")-1},
        {.role="assistant",.content="print(27)\n",.content_len=10},
        {.role="user",.content="<tool_response>\nok\n</tool_response>",
         .content_len=sizeof("<tool_response>\nok\n</tool_response>")-1},
        {.role="assistant",.content="created",.content_len=7},
        {.role="user",.content="task two",.content_len=8},
        {.role="assistant",.content="answer two",.content_len=10},
        {.role="user",.content="task three",.content_len=10},
    };
    CHECK(q27_agent_compaction_cut(raw_history, 10, 2, &cut) && cut == 7,
          "compaction keeps raw payload requests inside their tool root turn");

    char *snap2_path=NULL,*snap2_name=NULL;
    CHECK(q27_agent_session_new_snapshot_path(manifest,&snap2_path,&snap2_name,error,sizeof(error)) &&
          write_private(snap2_path,"Q27SNAP1-new",12),"replacement snapshot created");
    CHECK(q27_agent_session_publish(manifest,snap2_path,snap2_name,snap_name,
                                    messages,3,0,1,8192,tokenizer_sha1,
                                    snapshot_sha256,error,sizeof(error)) == 1,
          "replacement manifest publishes");
    CHECK(q27_agent_session_publish(manifest,snap2_path,snap2_name,snap_name,
                                    messages,3,0,1,8192,tokenizer_sha1,
                                    snapshot_sha256,error,sizeof(error)) == 0 &&
          strstr(error,"changed since"),
          "stale loaded writer cannot overwrite a newer manifest");
    CHECK(access(snap_path,F_OK)!=0 && errno==ENOENT,"old snapshot removed only after commit");
    CHECK(q27_agent_session_load(manifest,&loaded,error,sizeof(error)) &&
          !strcmp(loaded.snapshot_name,snap2_name),"replacement manifest loads");
    q27_agent_saved_session_free(&loaded);

    char *snap3_path=NULL,*snap3_name=NULL;
    CHECK(q27_agent_session_new_snapshot_path(manifest,&snap3_path,&snap3_name,error,sizeof(error)) &&
          write_private(snap3_path,"Q27SNAP1-uncertain",18),
          "uncertain replacement snapshot created");
    CHECK(setenv("Q27_AGENT_TEST_DIR_FSYNC_FAIL","1",1)==0,
          "directory sync failpoint enabled");
    CHECK(q27_agent_session_publish(manifest,snap3_path,snap3_name,snap2_name,
                                    messages,3,0,1,8192,tokenizer_sha1,
                                    snapshot_sha256,error,sizeof(error)) == 2,
          "post-rename directory failure reports uncertain commit");
    unsetenv("Q27_AGENT_TEST_DIR_FSYNC_FAIL");
    CHECK(access(snap3_path,F_OK)==0 && access(snap2_path,F_OK)==0,
          "uncertain commit retains both new and previous snapshots");
    CHECK(q27_agent_session_load(manifest,&loaded,error,sizeof(error)) &&
          !strcmp(loaded.snapshot_name,snap3_name),
          "visible uncertain manifest still references retained snapshot");
    q27_agent_saved_session_free(&loaded);

    int fd=open(manifest,O_RDWR); CHECK(fd>=0,"manifest opens for corruption");
    unsigned char byte; CHECK(pread(fd,&byte,1,40)==1,"manifest byte read");
    byte^=1; CHECK(pwrite(fd,&byte,1,40)==1 && close(fd)==0,"manifest corrupted");
    CHECK(!q27_agent_session_load(manifest,&loaded,error,sizeof(error)) && strstr(error,"corrupt"),
          "CRC corruption fails closed");

    CHECK(chmod(snap3_path,0644)==0,"snapshot mode weakened");
    CHECK(!q27_agent_session_publish(manifest,snap3_path,snap3_name,NULL,
                                     messages,3,0,1,8192,tokenizer_sha1,
                                     snapshot_sha256,error,sizeof(error)),
          "non-private snapshot rejected");
    chmod(snap3_path,0600);

    char lock_path[512];
    snprintf(lock_path,sizeof(lock_path),"%s/.chat.q27agent.lock",root);
    unlink(manifest); unlink(snap2_path); unlink(snap3_path); unlink(lock_path);
    rmdir(root);
    free(snap_path); free(snap_name); free(snap2_path); free(snap2_name);
    free(snap3_path); free(snap3_name);
    puts("q27 agent persistence selftest: PASS");
    return 0;
}
