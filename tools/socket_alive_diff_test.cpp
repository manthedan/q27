// Differential test: our socket_alive vs httplib::detail::is_socket_alive,
// on the states that matter. They must agree on every one.
#include "httplib.h"
#include <sys/socket.h>
#include <cstdio>
#include <cerrno>
#include <unistd.h>
static bool socket_alive(socket_t sock) {
    if (sock == INVALID_SOCKET) return true;
    fd_set fds; FD_ZERO(&fds); FD_SET(sock, &fds);
    timeval tv{0,0}; int r;
    do { r = select(static_cast<int>(sock+1), &fds, nullptr, nullptr, &tv); } while (r<0 && errno==EINTR);
    if (r == 0) return true;
    if (r < 0) return errno != EBADF;
    char b[1]; ssize_t n;
    do { n = recv(sock, b, sizeof(b), MSG_PEEK); } while (n<0 && errno==EINTR);
    return n > 0;
}
static int fails = 0;
static void cmp(const char* label, socket_t s) {
    bool a = socket_alive(s), b = httplib::detail::is_socket_alive(s);
    printf("  %-34s ours=%-5s httplib=%-5s %s\n", label, a?"true":"false", b?"true":"false",
           a==b ? "agree" : "DIVERGE");
    if (a != b) fails++;
}
int main(){
    int sv[2];
    // 1. open, idle, nothing written
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv); cmp("open, idle", sv[0]);
    // 2. peer wrote -> readable with data
    write(sv[1], "x", 1); cmp("peer wrote 1 byte", sv[0]);
    // 3. peer closed after data still buffered
    close(sv[1]); cmp("peer closed, data buffered", sv[0]);
    // 4. drain then peer-closed -> EOF
    char c; read(sv[0], &c, 1); cmp("peer closed, drained (EOF)", sv[0]);
    close(sv[0]);
    // 5. fully closed fd -> EBADF
    cmp("closed fd (EBADF)", sv[0]);
    // 6. fresh pair, peer half-closes write side
    socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    shutdown(sv[1], SHUT_WR); cmp("peer half-close (SHUT_WR)", sv[0]);
    close(sv[0]); close(sv[1]);
    printf("%s\n", fails ? "DIVERGENCE FOUND" : "identical on all states");
    return fails;
}
