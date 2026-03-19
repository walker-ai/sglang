from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Set, Union

import numpy as np
import torch

from sglang.srt.lora.eviction_policy import get_eviction_policy


DeltaBlob = Union[torch.Tensor, np.ndarray]


@dataclass
class DeltaBlobEntry:
    handle: int
    uid: Optional[str]
    tier: str  # "device" | "host" | "file"
    size_bytes: int
    blob_kind: str  # "torch" | "numpy"
    tensor: Optional[torch.Tensor] = None
    array: Optional[np.ndarray] = None
    file_path: Optional[str] = None


class DeltaBlobStore:
    """
    A 3-tier (device/host/file) blob store for delta-cache compressed payloads.

    Design goals:
    - Tiering is by LoRA uid (extra_key), consistent with MobiLoRA eviction policy.
    - On budget pressure, demote cold uids (LRU/FIFO) from device->host->file.
    - On access, optionally promote blobs back to device ("hot") to avoid repeated I/O.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        compression_backend: str,
        data_tier: str = "device",  # "device" | "host" | "file" | "auto"
        device_budget_bytes: Optional[int] = None,
        host_budget_bytes: Optional[int] = None,
        file_dir: Optional[str] = None,
        eviction_policy: str = "lru",
        host_pin_memory: bool = False,
        promote_on_get: bool = True,
        process_tag: Optional[str] = None,
    ) -> None:
        self.device = device
        self.compression_backend = compression_backend
        self.data_tier = data_tier
        self.device_budget_bytes = device_budget_bytes
        self.host_budget_bytes = host_budget_bytes
        self.host_pin_memory = host_pin_memory
        self.promote_on_get = promote_on_get

        if self.data_tier not in ("device", "host", "file", "auto"):
            raise ValueError(f"Unknown data_tier={data_tier!r}")

        self._lock = threading.Lock()
        self._entry_by_handle: Dict[int, DeltaBlobEntry] = {}
        self._handles_by_uid: Dict[Optional[str], Set[int]] = {}

        self._bytes_by_tier: Dict[str, int] = {"device": 0, "host": 0, "file": 0}

        self._eviction_policy = get_eviction_policy(eviction_policy)

        if file_dir is None:
            file_dir = os.path.join("sglang_storage", "delta_cache_blobs")
        if process_tag is None:
            process_tag = f"pid{os.getpid()}"
        self._file_dir = os.path.join(file_dir, process_tag)
        os.makedirs(self._file_dir, exist_ok=True)

    def bytes_in_tier(self, tier: str) -> int:
        return int(self._bytes_by_tier.get(tier, 0))

    def put(self, handle: int, uid: Optional[str], blob: DeltaBlob) -> None:
        with self._lock:
            if handle in self._entry_by_handle:
                raise KeyError(f"Duplicate handle {handle}")

            self._eviction_policy.mark_used(uid)

            entry = self._make_entry(handle=handle, uid=uid, blob=blob)
            self._entry_by_handle[handle] = entry
            self._handles_by_uid.setdefault(uid, set()).add(handle)

            self._bytes_by_tier[entry.tier] += entry.size_bytes
            self._maybe_demote_locked()

    def get(self, handle: int, *, want_device: bool) -> DeltaBlob:
        with self._lock:
            entry = self._entry_by_handle[handle]
            self._eviction_policy.mark_used(entry.uid)

            if entry.blob_kind == "numpy":
                if entry.tier == "file":
                    entry.array = self._read_file_to_numpy(entry.file_path, entry.size_bytes)
                    try:
                        os.remove(entry.file_path)
                    except FileNotFoundError:
                        pass
                    entry.file_path = None
                    self._move_tier_locked(entry, "host")
                return entry.array

            # torch uint8 path (cuszp)
            if want_device and entry.tier != "device" and self.promote_on_get:
                self._promote_torch_to_device_locked(entry)
                self._maybe_demote_locked()

            if entry.tier == "file":
                cpu_tensor = self._read_file_to_cpu_uint8(entry.file_path, entry.size_bytes)
                if want_device:
                    return cpu_tensor.to(device=self.device, non_blocking=False)
                return cpu_tensor

            if entry.tier == "host":
                if want_device:
                    return entry.tensor.to(device=self.device, non_blocking=False)
                return entry.tensor

            # device
            return entry.tensor

    def release(self, handle: int) -> None:
        with self._lock:
            entry = self._entry_by_handle.pop(handle, None)
            if entry is None:
                return

            self._bytes_by_tier[entry.tier] -= entry.size_bytes
            if entry.tier == "file" and entry.file_path:
                try:
                    os.remove(entry.file_path)
                except FileNotFoundError:
                    pass

            handles = self._handles_by_uid.get(entry.uid)
            if handles is not None:
                handles.discard(handle)
                if not handles:
                    self._handles_by_uid.pop(entry.uid, None)
                    self._eviction_policy.remove(entry.uid)

    def _make_entry(self, *, handle: int, uid: Optional[str], blob: DeltaBlob) -> DeltaBlobEntry:
        if isinstance(blob, torch.Tensor):
            if blob.dtype != torch.uint8:
                raise ValueError(f"Expected torch.uint8 blob, got {blob.dtype}")
            size_bytes = int(blob.numel() * blob.element_size())
            desired = "device" if (self.data_tier in ("device", "auto") and blob.is_cuda) else "host"
            if self.data_tier == "file":
                desired = "file"
            if self.data_tier == "host":
                desired = "host"

            if desired == "device":
                return DeltaBlobEntry(
                    handle=handle,
                    uid=uid,
                    tier="device",
                    size_bytes=size_bytes,
                    blob_kind="torch",
                    tensor=blob.contiguous(),
                )
            if desired == "host":
                cpu = blob.detach().to(device="cpu", non_blocking=False).contiguous()
                if self.host_pin_memory:
                    pinned = torch.empty_like(cpu, pin_memory=True)
                    pinned.copy_(cpu, non_blocking=False)
                    cpu = pinned
                return DeltaBlobEntry(
                    handle=handle,
                    uid=uid,
                    tier="host",
                    size_bytes=size_bytes,
                    blob_kind="torch",
                    tensor=cpu,
                )

            file_path = self._write_bytes_to_file(handle, blob.detach().contiguous().view(torch.uint8))
            return DeltaBlobEntry(
                handle=handle,
                uid=uid,
                tier="file",
                size_bytes=size_bytes,
                blob_kind="torch",
                file_path=file_path,
            )

        if isinstance(blob, np.ndarray):
            if blob.dtype != np.uint8:
                blob = blob.view(np.uint8)
            size_bytes = int(blob.nbytes)
            desired = "host" if self.data_tier in ("device", "host", "auto") else "file"
            if self.data_tier == "file":
                desired = "file"

            if desired == "host":
                return DeltaBlobEntry(
                    handle=handle,
                    uid=uid,
                    tier="host",
                    size_bytes=size_bytes,
                    blob_kind="numpy",
                    array=blob,
                )

            file_path = self._write_bytes_to_file(handle, blob)
            return DeltaBlobEntry(
                handle=handle,
                uid=uid,
                tier="file",
                size_bytes=size_bytes,
                blob_kind="numpy",
                file_path=file_path,
            )

        raise TypeError(f"Unsupported blob type: {type(blob).__name__}")

    def _maybe_demote_locked(self) -> None:
        if self.data_tier != "auto":
            return

        if self.device_budget_bytes is not None:
            while self._bytes_by_tier["device"] > self.device_budget_bytes:
                victim = self._select_victim_uid_locked("device")
                if victim is None:
                    break
                self._demote_uid_locked(victim, "device")

        if self.host_budget_bytes is not None:
            while self._bytes_by_tier["host"] > self.host_budget_bytes:
                victim = self._select_victim_uid_locked("host")
                if victim is None:
                    break
                self._demote_uid_locked(victim, "host")

    def _select_victim_uid_locked(self, tier: str) -> Optional[str]:
        candidates: Set[Optional[str]] = set()
        for uid, handles in self._handles_by_uid.items():
            for h in handles:
                if self._entry_by_handle[h].tier == tier:
                    candidates.add(uid)
                    break
        if not candidates:
            return None
        return self._eviction_policy.select_victim(candidates)

    def _demote_uid_locked(self, uid: Optional[str], tier: str) -> None:
        handles = self._handles_by_uid.get(uid, set())
        for h in list(handles):
            entry = self._entry_by_handle.get(h)
            if entry is None or entry.tier != tier:
                continue
            if tier == "device":
                if entry.blob_kind == "torch":
                    cpu = entry.tensor.detach().to(device="cpu", non_blocking=False).contiguous()
                    entry.tensor = cpu
                    self._move_tier_locked(entry, "host")
                else:
                    self._move_tier_locked(entry, "host")
            elif tier == "host":
                file_path = self._materialize_to_file_locked(entry)
                entry.tensor = None
                entry.array = None
                entry.file_path = file_path
                self._move_tier_locked(entry, "file")

    def _materialize_to_file_locked(self, entry: DeltaBlobEntry) -> str:
        if entry.blob_kind == "torch":
            if entry.tensor is None:
                raise RuntimeError("Missing host tensor for file materialization")
            data = entry.tensor.contiguous().view(torch.uint8)
            return self._write_bytes_to_file(entry.handle, data)
        if entry.blob_kind == "numpy":
            if entry.array is None:
                raise RuntimeError("Missing host array for file materialization")
            return self._write_bytes_to_file(entry.handle, entry.array)
        raise RuntimeError(f"Unknown blob_kind={entry.blob_kind!r}")

    def _promote_torch_to_device_locked(self, entry: DeltaBlobEntry) -> None:
        if entry.blob_kind != "torch":
            return
        if entry.tier == "device":
            return

        if entry.tier == "file":
            cpu_tensor = self._read_file_to_cpu_uint8(entry.file_path, entry.size_bytes)
            try:
                os.remove(entry.file_path)
            except FileNotFoundError:
                pass
            entry.file_path = None
            entry.tensor = cpu_tensor
            self._move_tier_locked(entry, "host")

        assert entry.tier == "host"
        entry.tensor = entry.tensor.to(device=self.device, non_blocking=False).contiguous()
        self._move_tier_locked(entry, "device")

    def _move_tier_locked(self, entry: DeltaBlobEntry, new_tier: str) -> None:
        if entry.tier == new_tier:
            return
        self._bytes_by_tier[entry.tier] -= entry.size_bytes
        self._bytes_by_tier[new_tier] += entry.size_bytes
        entry.tier = new_tier

    def _handle_to_path(self, handle: int) -> str:
        return os.path.join(self._file_dir, f"{handle}.bin")

    def _write_bytes_to_file(self, handle: int, data: Union[torch.Tensor, np.ndarray]) -> str:
        path = self._handle_to_path(handle)
        if isinstance(data, torch.Tensor):
            cpu = data.detach().to(device="cpu", non_blocking=False).contiguous()
            cpu.view(torch.uint8).numpy().tofile(path)
        else:
            np.asarray(data, dtype=np.uint8).tofile(path)
        return path

    @staticmethod
    def _read_file_to_cpu_uint8(path: str, size_bytes: int) -> torch.Tensor:
        buf = torch.empty((size_bytes,), dtype=torch.uint8, device="cpu")
        expected = buf.numel() * buf.element_size()
        with open(path, "rb", buffering=0) as f:
            mv = memoryview(buf.numpy())
            if f.readinto(mv) != expected:
                raise IOError(f"Short read: {path}")
        return buf

    @staticmethod
    def _read_file_to_numpy(path: str, size_bytes: int) -> np.ndarray:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) != size_bytes:
            raise IOError(f"Short read: {path}")
        return np.frombuffer(data, dtype=np.uint8).copy()
