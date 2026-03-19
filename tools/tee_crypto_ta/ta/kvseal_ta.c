#include <stdint.h>
#include <string.h>

#include <tee_internal_api.h>
#include <tee_internal_api_extensions.h>

#include "kvseal_ta.h"

#define STREAM_KEY_SIZE 16

static const uint8_t k_stream_key_obj_id[] = "kvseal_soft_stream_key_v1";

static TEE_Result load_or_create_secret(const uint8_t *obj_id, uint32_t obj_id_len,
                                        uint8_t *key, uint32_t key_len) {
    TEE_Result res;
    TEE_ObjectHandle obj = TEE_HANDLE_NULL;
    uint32_t read_sz = 0;

    res = TEE_OpenPersistentObject(TEE_STORAGE_PRIVATE, obj_id, obj_id_len,
                                   TEE_DATA_FLAG_ACCESS_READ, &obj);
    if (res == TEE_ERROR_ITEM_NOT_FOUND) {
        TEE_GenerateRandom(key, key_len);
        res = TEE_CreatePersistentObject(
            TEE_STORAGE_PRIVATE, obj_id, obj_id_len,
            TEE_DATA_FLAG_ACCESS_READ | TEE_DATA_FLAG_ACCESS_WRITE |
                TEE_DATA_FLAG_ACCESS_WRITE_META,
            TEE_HANDLE_NULL, key, key_len, &obj);
        if (res != TEE_SUCCESS) {
            EMSG("TEE_CreatePersistentObject failed: 0x%x", res);
            return res;
        }
        TEE_CloseObject(obj);
        return TEE_SUCCESS;
    }
    if (res != TEE_SUCCESS) {
        EMSG("TEE_OpenPersistentObject failed: 0x%x", res);
        return res;
    }

    res = TEE_ReadObjectData(obj, key, key_len, &read_sz);
    TEE_CloseObject(obj);
    if (res != TEE_SUCCESS) {
        EMSG("TEE_ReadObjectData failed: 0x%x", res);
        return res;
    }
    if (read_sz != key_len) {
        EMSG("persistent key size mismatch: %u", read_sz);
        return TEE_ERROR_CORRUPT_OBJECT;
    }
    return TEE_SUCCESS;
}

static uint32_t load32_le(const uint8_t *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void store32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

static void xtea_encrypt_block(uint32_t v[2], const uint32_t k[4]) {
    uint32_t v0 = v[0];
    uint32_t v1 = v[1];
    uint32_t sum = 0;
    const uint32_t delta = 0x9E3779B9;
    size_t i = 0;

    for (i = 0; i < 32; ++i) {
        v0 += (((v1 << 4) ^ (v1 >> 5)) + v1) ^ (sum + k[sum & 3]);
        sum += delta;
        v1 += (((v0 << 4) ^ (v0 >> 5)) + v0) ^ (sum + k[(sum >> 11) & 3]);
    }

    v[0] = v0;
    v[1] = v1;
}

static void fill_keystream_block(uint8_t out[8], const uint8_t key[STREAM_KEY_SIZE],
                                 uint64_t counter) {
    uint32_t k[4];
    uint32_t block[2];

    k[0] = load32_le(key + 0);
    k[1] = load32_le(key + 4);
    k[2] = load32_le(key + 8);
    k[3] = load32_le(key + 12);

    block[0] = (uint32_t)(counter & 0xffffffffu);
    block[1] = (uint32_t)(counter >> 32);
    xtea_encrypt_block(block, k);

    store32_le(out + 0, block[0]);
    store32_le(out + 4, block[1]);
}

static void soft_stream_xor(uint8_t *buf, uint32_t len, const uint8_t key[STREAM_KEY_SIZE]) {
    uint8_t ks[8];
    uint64_t counter = 0;
    uint32_t off = 0;

    while (off < len) {
        uint32_t i = 0;
        uint32_t chunk = len - off;
        if (chunk > sizeof(ks)) {
            chunk = sizeof(ks);
        }

        fill_keystream_block(ks, key, counter++);
        for (i = 0; i < chunk; ++i) {
            buf[off + i] ^= ks[i];
        }
        off += chunk;
    }
}

static TEE_Result cmd_ping(TEE_Param params[4]) {
    params[1].value.a = 42;
    return TEE_SUCCESS;
}

/*
 * Reuse the existing helper command name "hmac" as a temporary raw
 * encrypt/decrypt entry point so we don't have to change host code.
 * The transform is symmetric: running the same command twice restores input.
 */
static TEE_Result cmd_soft_stream_raw(TEE_Param params[4]) {
    TEE_Result res;
    uint8_t key[STREAM_KEY_SIZE];
    uint8_t *buf = (uint8_t *)params[0].memref.buffer;
    uint32_t cap = params[0].memref.size;
    uint32_t in_len = params[1].value.a;

    if (!buf || in_len > cap) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    res = load_or_create_secret(k_stream_key_obj_id, sizeof(k_stream_key_obj_id),
                                key, sizeof(key));
    if (res != TEE_SUCCESS) {
        return res;
    }

    soft_stream_xor(buf, in_len, key);
    params[1].value.a = in_len;
    return TEE_SUCCESS;
}

static TEE_Result cmd_stream_selftest(TEE_Param params[4]) {
    static const uint8_t key[STREAM_KEY_SIZE] = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
        0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
    };
    uint8_t buf[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
    };
    uint8_t orig[16];

    (void)params;
    TEE_MemMove(orig, buf, sizeof(buf));
    soft_stream_xor(buf, sizeof(buf), key);
    soft_stream_xor(buf, sizeof(buf), key);

    if (TEE_MemCompare(buf, orig, sizeof(buf)) != 0) {
        return TEE_ERROR_SECURITY;
    }
    return TEE_SUCCESS;
}

TEE_Result TA_CreateEntryPoint(void) {
    return TEE_SUCCESS;
}

void TA_DestroyEntryPoint(void) {
}

TEE_Result TA_OpenSessionEntryPoint(uint32_t param_types, TEE_Param params[4], void **sess_ctx) {
    (void)param_types;
    (void)params;
    (void)sess_ctx;
    return TEE_SUCCESS;
}

void TA_CloseSessionEntryPoint(void *sess_ctx) {
    (void)sess_ctx;
}

TEE_Result TA_InvokeCommandEntryPoint(void *sess_ctx, uint32_t cmd_id, uint32_t param_types,
                                      TEE_Param params[4]) {
    (void)sess_ctx;

    if (param_types != TEE_PARAM_TYPES(TEE_PARAM_TYPE_MEMREF_INOUT,
                                       TEE_PARAM_TYPE_VALUE_INOUT,
                                       TEE_PARAM_TYPE_NONE,
                                       TEE_PARAM_TYPE_NONE)) {
        return TEE_ERROR_BAD_PARAMETERS;
    }

    switch (cmd_id) {
        case CMD_PING:
            return cmd_ping(params);
        case CMD_HMAC_RAW:
            return cmd_soft_stream_raw(params);
        case CMD_AES_SELFTEST:
            return cmd_stream_selftest(params);
        default:
            return TEE_ERROR_NOT_SUPPORTED;
    }
}
