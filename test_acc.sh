#! /bin/bash

proxy
python benchmark/delta_cache/test_acc.py --base baseline.jsonl --delta delta_cache.jsonl
