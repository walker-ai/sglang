#ifndef USER_TA_HEADER_DEFINES_H
#define USER_TA_HEADER_DEFINES_H

#include "../include/kvseal_ta.h"

#define TA_UUID             KVSEAL_TA_UUID
#define TA_FLAGS            TA_FLAG_EXEC_DDR
#define TA_STACK_SIZE       (4 * 1024)
#define TA_DATA_SIZE        (64 * 1024)
#define TA_VERSION          "1.0"
#define TA_DESCRIPTION      "KV seal/unseal TA (AES-GCM)"

#endif /* USER_TA_HEADER_DEFINES_H */
