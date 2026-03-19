# tee_crypto_ta

AES-256-GCM seal/unseal TA for compressed KV blob simulation.

## Layout

- `include/kvseal_ta.h`: UUID + command IDs + format constants
- `ta/`: Trusted Application (AES-GCM)
- `host/tee_helper.c`: CA CLI bridge (`stdin -> TA -> stdout`)

## Build TA (x86 container cross compile)

```bash
make -C ta \
  CROSS_COMPILE="<toolchain>/bin/aarch64-buildroot-linux-gnu-" \
  TA_DEV_KIT_DIR="<optee>/build/t234/export-ta_arm64" \
  -j"$(nproc)"
```

Deploy TA on Jetson:

```bash
sudo cp ta/6f6fbe10-4ad1-4d3e-b4be-5f96f14b6d49.ta /lib/optee_armtz/
```

## Build helper on Jetson

```bash
make -C host
```

(or cross compile helper)

```bash
make -C host \
  CROSS_COMPILE="<toolchain>/bin/aarch64-buildroot-linux-gnu-" \
  TEEC_EXPORT="<optee>/install/t234/usr"
```

## Smoke test

```bash
echo -n "abcd" | host/tee_helper seal | host/tee_helper unseal
```

## Use with demo script

```bash
PYTHONPATH=python python python/tee_delta_cuszp_seal_demo.py \
  --helper-cmd "<abs_path>/tools/tee_crypto_ta/host/tee_helper"
```

