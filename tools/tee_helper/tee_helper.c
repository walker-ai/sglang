#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <tee_client_api.h>

/*
 * NOTE:
 * 1) Update UUID / CMD ids to match your TA implementation.
 * 2) CLI:
 *      tee_helper seal   < plain.bin > sealed.bin
 *      tee_helper unseal < sealed.bin > plain.bin
 */

/* 8aaaf200-2450-11e4-abe2-0002a5d5c51b */
static const TEEC_UUID kTaUuid = {
    0x8aaaf200, 0x2450, 0x11e4,
    {0xab, 0xe2, 0x00, 0x02, 0xa5, 0xd5, 0xc5, 0x1b},
};

/* Replace with your TA command IDs */
#define CMD_SEAL   2
#define CMD_UNSEAL 3

static uint8_t *read_all_stdin(size_t *out_size) {
    size_t cap = 1 << 20;
    size_t n = 0;
    uint8_t *buf = (uint8_t *)malloc(cap);
    if (!buf) return NULL;

    for (;;) {
        if (n == cap) {
            size_t new_cap = cap * 2;
            uint8_t *tmp = (uint8_t *)realloc(buf, new_cap);
            if (!tmp) {
                free(buf);
                return NULL;
            }
            buf = tmp;
            cap = new_cap;
        }
        size_t r = fread(buf + n, 1, cap - n, stdin);
        n += r;
        if (r == 0) {
            if (feof(stdin)) break;
            free(buf);
            return NULL;
        }
    }
    *out_size = n;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s seal|unseal\n", argv[0]);
        return 2;
    }

    uint32_t cmd = 0;
    if (strcmp(argv[1], "seal") == 0) {
        cmd = CMD_SEAL;
    } else if (strcmp(argv[1], "unseal") == 0) {
        cmd = CMD_UNSEAL;
    } else {
        fprintf(stderr, "Unknown op: %s\n", argv[1]);
        return 2;
    }

    size_t in_size = 0;
    uint8_t *in = read_all_stdin(&in_size);
    if (!in) {
        fprintf(stderr, "Failed to read stdin\n");
        return 1;
    }

    /* Reserve extra room for AEAD overhead in seal path. */
    size_t out_cap = in_size + 128;
    if (out_cap < 256) out_cap = 256;
    uint8_t *io = (uint8_t *)malloc(out_cap);
    if (!io) {
        free(in);
        fprintf(stderr, "OOM\n");
        return 1;
    }
    memcpy(io, in, in_size);
    free(in);

    TEEC_Context ctx;
    TEEC_Session sess;
    TEEC_Operation op;
    uint32_t origin = 0;
    TEEC_Result res;

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INOUT, TEEC_NONE, TEEC_NONE, TEEC_NONE);
    op.params[0].tmpref.buffer = io;
    op.params[0].tmpref.size = out_cap;

    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_InitializeContext failed: 0x%x\n", res);
        free(io);
        return 1;
    }

    res = TEEC_OpenSession(
        &ctx, &sess, &kTaUuid, TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_OpenSession failed: 0x%x origin=0x%x\n", res, origin);
        TEEC_FinalizeContext(&ctx);
        free(io);
        return 1;
    }

    res = TEEC_InvokeCommand(&sess, cmd, &op, &origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_InvokeCommand failed: 0x%x origin=0x%x\n", res, origin);
        TEEC_CloseSession(&sess);
        TEEC_FinalizeContext(&ctx);
        free(io);
        return 1;
    }

    size_t out_size = op.params[0].tmpref.size;
    if (out_size > out_cap) {
        fprintf(stderr, "TA returned invalid output size: %zu > %zu\n", out_size, out_cap);
        TEEC_CloseSession(&sess);
        TEEC_FinalizeContext(&ctx);
        free(io);
        return 1;
    }

    if (fwrite(io, 1, out_size, stdout) != out_size) {
        fprintf(stderr, "Failed to write stdout\n");
        TEEC_CloseSession(&sess);
        TEEC_FinalizeContext(&ctx);
        free(io);
        return 1;
    }

    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
    free(io);
    return 0;
}

