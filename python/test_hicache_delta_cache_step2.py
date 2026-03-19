from pathlib import Path


def test_hiradix_cache_wires_delta_cache_into_radix_init():
    repo_python_root = Path(__file__).resolve().parent
    path = repo_python_root / "sglang" / "srt" / "mem_cache" / "hiradix_cache.py"
    text = path.read_text(encoding="utf-8")

    assert "enable_delta_cache=enable_delta_cache" in text
    assert "compression_backend=compression_backend" in text


def test_hiradix_cache_match_prefix_has_delta_branch_and_reconstruct():
    repo_python_root = Path(__file__).resolve().parent
    path = repo_python_root / "sglang" / "srt" / "mem_cache" / "hiradix_cache.py"
    text = path.read_text(encoding="utf-8")

    assert "if not self.enable_delta_cache" in text
    assert "_reconstruct_and_alloc" in text


def test_hiradix_cache_delta_page_size_guard_present():
    repo_python_root = Path(__file__).resolve().parent
    path = repo_python_root / "sglang" / "srt" / "mem_cache" / "hiradix_cache.py"
    text = path.read_text(encoding="utf-8")

    assert "delta-cache currently only supports page_size=1" in text

