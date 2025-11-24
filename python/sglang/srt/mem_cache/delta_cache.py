# =============================================================================
# Compression Backend Strategy
# =============================================================================
from typing import Dict, Union, Tuple, Optional, List, Iterator

import torch
import ctypes, sys

import numpy as np

class CompressorBackend:
    def compress(self, tensor: torch.Tensor, error_bound: float) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError

    def decompress(self, data: Union[torch.Tensor, np.ndarray], shape: torch.Size, dtype: torch.dtype, error_bound: float) -> torch.Tensor:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError
    
class SZBackend(CompressorBackend):
    def __init__(self):
        from sglang.srt.lora_diff.pysz import SZ
        lib_extension = {
            "darwin": "libSZ3c.dylib",
            "windows": "SZ3c.dll",
        }.get(sys.platform, "libSZ3c.so")
        # 请根据实际路径调整
        self.sz = SZ(f"/home/wangyitao/tools/SZ3/install/lib/{lib_extension}")
        print("[RadixCache] Initialized SZ Backend (CPU)")

    def compress(self, target_tensor: torch.Tensor, error_bound: float) -> np.ndarray:
        # SZ 需要 numpy 输入 (CPU)
        diff_numpy = target_tensor.detach().cpu().float().numpy()
        diff_compressed, _ = self.sz.compress(diff_numpy, eb_mode=0, eb_abs=error_bound, eb_rel=0, eb_pwr=0)
        return diff_compressed

    def decompress(self, compressed_data: np.ndarray, shape: torch.Size, dtype: torch.dtype, error_bound: float) -> torch.Tensor:
        # SZ 解压不需要 error_bound (元数据在 header 中)，但为了接口统一保留参数
        decompressed_numpy = self.sz.decompress(compressed_data, shape, original_dtype=np.float32)
        # 转回 Tensor 并移动到 GPU
        return torch.from_numpy(decompressed_numpy).to("cuda").to(dtype)

    def get_name(self) -> str:
        return "sz"

class CuSZpBackend(CompressorBackend):
    def __init__(self, device):
        from sglang.srt.lora_diff.pycuSZp import cuSZp
        self.cusz = cuSZp()
        self.device = device
        # 常量定义
        self.CUSZ_MODE_ABS = 1
        self.CUSZ_DTYPE_FP32 = 0
        print("[RadixCache] Initialized cuSZp Backend (GPU)")

    def compress(self, target_tensor: torch.Tensor, error_bound: float) -> torch.Tensor:
        # 1. 转换类型并确保内存连续 (GPU->GPU copy)
        src_float = target_tensor.to(torch.float32).contiguous()
        
        # 2. 预分配 Buffer
        max_size = src_float.numel() * src_float.element_size()
        compressed_buffer = torch.empty(max_size, dtype=torch.uint8, device=self.device).contiguous()
        num_ele = src_float.numel()
        
        # 3. 调用 CUDA 压缩
        compressed_len = self.cusz.compress(
            d_oriData=ctypes.c_void_p(src_float.data_ptr()),
            d_cmpBytes=ctypes.c_void_p(compressed_buffer.data_ptr()),
            num_elements=num_ele,
            error_bound=error_bound,
            dim=1,
            dims=(num_ele, 1, 1),
            data_type=self.CUSZ_DTYPE_FP32,
            mode=self.CUSZ_MODE_ABS
        )

        # 4. 安全 Padding (防止 Warp Illegal Address)
        SAFETY_PADDING = 4096 
        padded_len = compressed_len + SAFETY_PADDING
        if padded_len > max_size:
            padded_len = max_size

        return compressed_buffer[:padded_len].clone()

    def decompress(self, compressed_data: torch.Tensor, shape: torch.Size, dtype: torch.dtype, error_bound: float) -> torch.Tensor:
        num_elements = 1
        for dim in shape:
            num_elements *= dim
            
        decompressed_float = torch.empty(num_elements, dtype=torch.float32, device=self.device).contiguous()
        compressed_len = compressed_data.numel() # 包含 Padding 的长度

        self.cusz.decompress(
            d_decData=ctypes.c_void_p(decompressed_float.data_ptr()),
            d_cmpBytes=ctypes.c_void_p(compressed_data.data_ptr()),
            num_elements=num_elements,
            compressed_size=compressed_len, 
            error_bound=error_bound,
            dim=1,
            dims=(num_elements, 1, 1),
            data_type=self.CUSZ_DTYPE_FP32,
            mode=self.CUSZ_MODE_ABS 
        )
        
        return decompressed_float.to(dtype).view(shape)

    def get_name(self) -> str:
        return "cuszp"