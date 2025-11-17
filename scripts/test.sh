#! /bin/bash

python -m sglang.launch_server --model-path=/AI/HF_MODELS/Qwen3-8B --port 8188 --host 0.0.0.0 --enable-delta-cache --max-running-requests 1
