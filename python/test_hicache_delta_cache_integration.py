from pathlib import Path


def test_hiradix_cache_has_actionable_delta_cache_guard():
    # NOTE: This is a source-level regression test (no runtime import) so it can run in
    # minimal environments where importing the full `sglang` package is not possible.
    repo_python_root = Path(__file__).resolve().parent
    hiradix_cache_path = (
        repo_python_root / "sglang" / "srt" / "mem_cache" / "hiradix_cache.py"
    )
    text = hiradix_cache_path.read_text(encoding="utf-8")

    assert "enable_delta_cache" in text
    assert "delta-cache currently only supports page_size=1" in text


def test_scheduler_wires_delta_cache_args_into_hiradix_cache():
    repo_python_root = Path(__file__).resolve().parent
    scheduler_path = repo_python_root / "sglang" / "srt" / "managers" / "scheduler.py"
    text = scheduler_path.read_text(encoding="utf-8")

    assert "HiRadixCache(" in text
    assert "enable_delta_cache=server_args.enable_delta_cache" in text
    assert "compression_backend=server_args.compression_backend" in text
