import os
import shutil

import numpy as np
import torch

from sglang.srt.mem_cache.delta_blob_store import DeltaBlobStore


def test_delta_blob_store_host_to_file_tiering():
    base_dir = os.path.join(os.path.dirname(__file__), "_tmp_delta_blob_store")
    shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)

    store = DeltaBlobStore(
        device=torch.device("cpu"),
        compression_backend="sz",
        data_tier="auto",
        device_budget_bytes=0,
        host_budget_bytes=10,
        file_dir=base_dir,
        eviction_policy="lru",
        process_tag="test",
    )

    blob_a = np.arange(8, dtype=np.uint8)
    blob_b = np.arange(8, dtype=np.uint8) + 10

    store.put(1, "lora_a", blob_a)
    store.put(2, "lora_b", blob_b)

    # Host budget=10, so one uid should be demoted to file.
    assert store.bytes_in_tier("host") <= 10
    assert store.bytes_in_tier("file") >= 8

    got = store.get(1, want_device=False)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.uint8

    store.release(1)
    store.release(2)
    shutil.rmtree(base_dir, ignore_errors=True)

