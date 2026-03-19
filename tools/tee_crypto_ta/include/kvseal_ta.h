#ifndef KVSEAL_TA_H
#define KVSEAL_TA_H

/* 9d1c7b52-5d33-4b7e-9d3a-51c2bb7c1f21 */
#define KVSEAL_TA_UUID \
    { 0x9d1c7b52, 0x5d33, 0x4b7e, \
      { 0x9d, 0x3a, 0x51, 0xc2, 0xbb, 0x7c, 0x1f, 0x21 } }

#define CMD_PING         1
#define CMD_HMAC_RAW     2
#define CMD_AES_SELFTEST 3

#define KVSEAL_TAG_LEN 32U

#endif /* KVSEAL_TA_H */
