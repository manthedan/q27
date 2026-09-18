#ifndef Q27_AGENT_SHA256_H
#define Q27_AGENT_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    unsigned char block[64];
    size_t used;
} q27_agent_sha256_ctx;

void q27_agent_sha256_init(q27_agent_sha256_ctx *ctx);
void q27_agent_sha256_update(q27_agent_sha256_ctx *ctx,
                             const unsigned char *data, size_t len);
void q27_agent_sha256_final(q27_agent_sha256_ctx *ctx,
                            unsigned char digest[32]);
void q27_agent_sha256(const unsigned char *data, size_t len,
                      unsigned char digest[32]);

#endif
