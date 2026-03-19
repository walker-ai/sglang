import torch
# import pycuda.driver as pycuda
# import pycuda.autoinit  # Automatically initializes CUDA driver
from .pycuSZp import cuSZp
import ctypes

import time

compressor = cuSZp()
data_type_flag = 0  # 0 表示 float32，目前只支持 float32 和 float64

def compress(seed_tensor, target_tensor):
    # 检查形状一致性
    if seed_tensor.shape != target_tensor.shape:
        raise ValueError("seed_tensor and target_tensor must have the same shape")

    # 计算余弦相似度
    cos_sim = cosine_similarity(seed_tensor, target_tensor)
    if cos_sim > 0.975:
        eb_abs = 1e-4
    elif cos_sim > 0.85:
        eb_abs = 1e-3
    else:
        eb_abs = 1e-2

        
    # 计算差分
    diff_tensor = (target_tensor - seed_tensor).to(torch.float32).contiguous()

    assert diff_tensor.is_contiguous(), "差分张量内存不连续"

    diff_tensor_size = diff_tensor.numel() * diff_tensor.element_size()

    # 使用 cuszp 进行压缩
    # compressed_buffer = pycuda.mem_alloc(diff_tensor_size)
    
    # compressed_buffer_size = int(diff_tensor.numel() * 4 * 100)  # float32 → 4字节/元素
    compressed_buffer = torch.empty(diff_tensor_size, dtype=torch.uint8, device='cuda').contiguous()

    compressed_size = compressor.compress(
        ctypes.c_void_p(diff_tensor.data_ptr()),
        ctypes.c_void_p(compressed_buffer.data_ptr()),
        diff_tensor.numel(),
        eb_abs,
        data_type=data_type_flag,
        mode=0
    )



    assert compressed_size <= compressed_buffer.size(0), \
        f"Buffer overflow: {compressed_size} > {compressed_buffer.size(0)}"
    
    # 只保留实际压缩后的数据
    with open("/home/orin/workspace/paper/mobilora/log/mobilora_output_4_S5_xsum.txt", "a") as f:
        print(f"diff_tensor_size = {diff_tensor_size}, compressed_size = {compressed_size}, compress_ratio = {(compressed_size / diff_tensor_size * 100):.5f}%\n")
    
    time.sleep(1)

    # print(f'len(compressed_buffer) = {len(compressed_buffer)}, compressed_buffer.shape={compressed_buffer.shape}, compressed_size = {compressed_size}\n')
    compressed_data = compressed_buffer[:compressed_size].clone()

    return compressed_data, compressed_data.shape, eb_abs, diff_tensor


def decompress(seed_tensor, target_tensor_diff, eb_abs, origin_diff_tensor) -> torch.Tensor:
    """
    解压缩函数：从基础张量和差分数据恢复目标张量
    """
    # 创建解压输出缓冲区
    d_decData = torch.empty_like(origin_diff_tensor, device='cuda')
    origin_data = target_tensor_diff.clone()
    
    compressor.decompress(
        ctypes.c_void_p(d_decData.data_ptr()),
        ctypes.c_void_p(origin_data.data_ptr()),
        origin_diff_tensor.numel(),
        len(target_tensor_diff),
        eb_abs,
        data_type=data_type_flag,
        mode=0
    )
    # 恢复目标张量
    d_decData = d_decData.to(torch.float16)

    print(f"Decompressed tensor matches original within error bound: {torch.allclose(origin_diff_tensor.half(), d_decData, atol=1E-2)}")
    target_tensor_restored = seed_tensor + d_decData

    return target_tensor_restored