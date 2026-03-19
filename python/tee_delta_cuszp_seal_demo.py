#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import time

import torch

from sglang.srt.mem_cache.delta_cache import CuSZpBackend


def run_helper(helper_cmd: str, op: str, payload: bytes, timeout_s: float) -> bytes:
    argv = shlex.split(helper_cmd) + [op]
    proc = subprocess.run(
        argv,
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"helper failed: op={op}, code={proc.returncode}, err={err}")
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(
        description="Standalone demo: delta tensor -> cuSZp -> TEE encrypt/decrypt -> decompress verify"
    )
    parser.add_argument(
        "--helper-cmd",
        type=str,
        required=True,
        help="TEE helper command. It will be invoked as '<cmd> hmac' twice as encrypt/decrypt.",
    )
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--error-bound", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuSZp demo.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    shape = (args.num_tokens, args.num_heads, args.head_dim)
    base = torch.randn(shape, device=device, dtype=torch.float32)
    target = base + 1e-3 * torch.randn(shape, device=device, dtype=torch.float32)
    diff = target - base

    backend = CuSZpBackend(device=device)

    t0 = time.perf_counter()
    comp_tensor, comp_len = backend.compress(diff, error_bound=args.error_bound)
    t1 = time.perf_counter()

    comp_host = (
        comp_tensor[:comp_len]
        .detach()
        .to(device="cpu", non_blocking=False)
        .contiguous()
        .numpy()
        .tobytes()
    )

    t2 = time.perf_counter()
    sealed = run_helper(args.helper_cmd, "hmac", comp_host, args.timeout_s)
    t3 = time.perf_counter()
    unsealed = run_helper(args.helper_cmd, "hmac", sealed, args.timeout_s)
    t4 = time.perf_counter()

    if len(unsealed) != comp_len:
        raise RuntimeError(f"unsealed size mismatch: {len(unsealed)} != {comp_len}")

    restored_comp = torch.frombuffer(unsealed, dtype=torch.uint8).clone().to(device=device)
    diff_restored = backend.decompress(
        compressed_data=restored_comp,
        shape=diff.shape,
        dtype=diff.dtype,
        error_bound=args.error_bound,
        compressed_len=comp_len,
    )
    t5 = time.perf_counter()

    recon = base + diff_restored
    max_diff_err = (diff_restored - diff).abs().max().item()
    mean_diff_err = (diff_restored - diff).abs().mean().item()
    max_target_err = (recon - target).abs().max().item()
    mean_target_err = (recon - target).abs().mean().item()

    orig_bytes = diff.numel() * diff.element_size()
    ratio = (orig_bytes / comp_len) if comp_len > 0 else 0.0

    print("=== Delta cuSZp + TEE encrypt/decrypt demo ===")
    print(f"shape={shape}, error_bound={args.error_bound}")
    print(f"orig_bytes={orig_bytes}, comp_len={comp_len}, comp_ratio={ratio:.3f}x")
    print(f"sealed_bytes={len(sealed)}, overhead={len(sealed) - comp_len}")
    print(
        "time_ms: "
        f"compress={(t1-t0)*1000:.2f}, "
        f"encrypt={(t3-t2)*1000:.2f}, "
        f"decrypt={(t4-t3)*1000:.2f}, "
        f"decompress={(t5-t4)*1000:.2f}"
    )
    print(
        "error: "
        f"max_diff={max_diff_err:.6e}, mean_diff={mean_diff_err:.6e}, "
        f"max_target={max_target_err:.6e}, mean_target={mean_target_err:.6e}"
    )


if __name__ == "__main__":
    main()
