#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <tee_client_api.h>

#include "../include/kvseal_ta.h"

static uint8_t *read_all_stdin(size_t *out_size) {
    size_t cap = 1 << 20;
    size_t n = 0;
    uint8_t *buf = malloc(cap);
    if (!buf) {
        return NULL;
    }

    for (;;) {
        if (n == cap) {
            size_t new_cap = cap * 2;
            uint8_t *tmp = realloc(buf, new_cap);
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
            if (feof(stdin)) {
                break;
            }
            free(buf);
            return NULL;
        }
    }

    *out_size = n;
    return buf;
}

int main(int argc, char **argv) {
    TEEC_Context ctx;
    TEEC_Session sess;
    TEEC_Operation op;
    TEEC_Result res;
    uint32_t origin = 0;
    TEEC_UUID uuid = KVSEAL_TA_UUID;
    uint32_t cmd = 0;
    size_t in_len = 0;
    uint8_t *io = NULL;
    size_t out_cap = 0;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s ping|hmac|aes-selftest\n", argv[0]);
        return 2;
    }

    if (strcmp(argv[1], "ping") == 0) {
        cmd = CMD_PING;
        out_cap = 64;
        io = calloc(1, out_cap);
        if (!io) {
            return 1;
        }
    } else if (strcmp(argv[1], "hmac") == 0) {
        cmd = CMD_HMAC_RAW;
        io = read_all_stdin(&in_len);
        if (!io) {
            fprintf(stderr, "Failed to read stdin\n");
            return 1;
        }
        out_cap = in_len + KVSEAL_TAG_LEN + 64;
        uint8_t *tmp = realloc(io, out_cap);
        if (!tmp) {
            free(io);
            fprintf(stderr, "OOM\n");
            return 1;
        }
        io = tmp;
    } else if (strcmp(argv[1], "aes-selftest") == 0) {
        cmd = CMD_AES_SELFTEST;
        out_cap = 64;
        io = calloc(1, out_cap);
        if (!io) {
            return 1;
        }
    } else {
        fprintf(stderr, "Unknown op: %s\n", argv[1]);
        return 2;
    }

    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_InitializeContext failed: 0x%x\n", res);
        free(io);
        return 1;
    }

    res = TEEC_OpenSession(&ctx, &sess, &uuid, TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_OpenSession failed: 0x%x origin=0x%x\n", res, origin);
        TEEC_FinalizeContext(&ctx);
        free(io);
        return 1;
    }

    memset(&op, 0, sizeof(op));
    op.paramTypes = TEEC_PARAM_TYPES(
        TEEC_MEMREF_TEMP_INOUT,
        TEEC_VALUE_INOUT,
        TEEC_NONE,
        TEEC_NONE);
    op.params[0].tmpref.buffer = io;
    op.params[0].tmpref.size = out_cap;
    op.params[1].value.a = (uint32_t)in_len;

    res = TEEC_InvokeCommand(&sess, cmd, &op, &origin);
    if (res != TEEC_SUCCESS) {
        fprintf(stderr, "TEEC_InvokeCommand failed: 0x%x origin=0x%x\n", res, origin);
        TEEC_CloseSession(&sess);
        TEEC_FinalizeContext(&ctx);
        free(io);
        return 1;
    }

    if (cmd == CMD_PING) {
        fprintf(stderr, "ping=%u\n", op.params[1].value.a);
    } else if (cmd == CMD_AES_SELFTEST) {
        fprintf(stderr, "aes-selftest=ok\n");
    } else if (cmd == CMD_HMAC_RAW) {
        uint32_t out_len = op.params[1].value.a;
        if (fwrite(io, 1, out_len, stdout) != out_len) {
            fprintf(stderr, "Failed to write stdout\n");
            TEEC_CloseSession(&sess);
            TEEC_FinalizeContext(&ctx);
            free(io);
            return 1;
        }
    }

    TEEC_CloseSession(&sess);
    TEEC_FinalizeContext(&ctx);
    free(io);
    return 0;
}
