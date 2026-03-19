# tee_helper

Small CLI bridge for Python demo:

- stdin: binary payload
- argv[1]: `seal` or `unseal`
- stdout: binary result from TA

## Build on Jetson

```bash
cd tools/tee_helper
make
```

## Smoke test

```bash
echo -n "abcd" | ./tee_helper seal | ./tee_helper unseal
```

## Use with demo

```bash
PYTHONPATH=python python python/tee_delta_cuszp_seal_demo.py \
  --helper-cmd "/absolute/path/to/tools/tee_helper/tee_helper"
```

## Important

`tee_helper.c` currently uses:

- TA UUID: `8aaaf200-2450-11e4-abe2-0002a5d5c51b`
- `CMD_SEAL=2`
- `CMD_UNSEAL=3`

Update these constants to match your TA.

