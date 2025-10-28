import torch
import numpy as np
from numpy import ndarray
from sglang.srt.lora_diff.pysz import SZ
import time

# 初始化 SZ3
import sys
lib_extention = {
    "darwin": "libSZ3c.dylib",
    "windows": "SZ3c.dll",
}.get(sys.platform, "libSZ3c.so")
sz = SZ("/home/walker/workspace/SZ3/install/lib/{}".format(lib_extention))

def cosine_similarity(tensor1, tensor2) -> float:
    """
    按行计算两个张量的余弦相似度，然后取平均值
    """
    assert tensor1.shape == tensor2.shape, "Tensor shapes must match for cosine similarity calculation"
    cos_sim_per_row = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=1)
    return cos_sim_per_row.mean().item()

def compress(seed_tensor, target_tensor) -> tuple[ndarray, ndarray]:
    """
    压缩函数：计算目标张量与基础张量的差分，并进行压缩
    根据余弦相似度选择误差界限：
        - 相似度 > 0.975 时，选择 1e-4
        - 0.85 < 相似度 <= 0.975 时，选择 1e-3
        - 相似度 <= 0.85 时，选择 1e-2

    输入：seed_tensor, target_tensor, [32, seq_len, hidden_state]
    """
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
    diff_tensor = target_tensor - seed_tensor
    diff_tensor = diff_tensor.to(torch.float32)
    diff_numpy = diff_tensor.detach().cpu().numpy()  # 转为 NumPy 数组

    # 使用 sz.compress 进行压缩
    diff_compressed, _ = sz.compress(diff_numpy, eb_mode=0, eb_abs=eb_abs, eb_rel=0, eb_pwr=0)

    diff_tensor_size = diff_tensor.numel() * diff_tensor.element_size()
    compressed_size = diff_compressed.nbytes


    return diff_compressed, diff_tensor.shape, eb_abs, diff_tensor


def decompress(seed_tensor, target_tensor_diff, shape, dtype) -> torch.Tensor:
    """
    解压缩函数：从基础张量和差分数据恢复目标张量
    """
    # 解压缩差分
    diff_decompressed = sz.decompress(target_tensor_diff, shape, original_dtype=dtype)

    # 恢复目标张量
    diff_tensor_restored = torch.tensor(diff_decompressed, dtype=seed_tensor.dtype, device='cuda')
    target_tensor_restored = seed_tensor + diff_tensor_restored

    return target_tensor_restored


# 测试代码
if __name__ == "__main__":
    # 随机初始化两个张量
    tensor_size = (470, 32, 128)  # 设置一个示例大小
    
    
    base_tensors = []
    target_tensors = []
    diff_tensors = []
    for i in range(32):
        base_tensor = torch.randn(tensor_size)
        target_tensor = torch.randn(tensor_size)

        base_tensors.append(base_tensor)
        target_tensors.append(target_tensor)

        compressed_diff = compress(base_tensor, target_tensor)
        diff_tensors.append(compressed_diff)


    # decompress_time = time.time()
    restored_tensors = []
    for i in range(32):
        decompress_time = time.time()
        restored_tensor = decompress(
            base_tensors[i], diff_tensors[i], shape=tensor_size, dtype=np.float32
        )
        decompress_time = time.time() - decompress_time
        print(f"decompress time = {decompress_time}")

        restored_tensors.append(restored_tensor)
    # decompress_time = time.time() - decompress_time

    # print(f"decompress time = {decompress_time}")
  
    base_tensor = torch.randn(tensor_size)
    target_tensor = torch.randn(tensor_size)

    # 压缩目标张量
    print("Compressing...")
    compressed_diff, shape, eb_abs_used = compress(base_tensor, target_tensor)
    print(f"Compression complete. Compressed size: {len(compressed_diff)} bytes")
    print(f"Shape: {shape}, Error Bound Used: {eb_abs_used}")

    # 计算压缩前的差分数据大小
    diff_tensor = target_tensor - base_tensor
    diff_numpy = diff_tensor.detach().cpu().numpy()  # 转为 NumPy 数组
    original_size = diff_numpy.nbytes  # 获取原始差分数据大小

    # 计算压缩率
    compressed_size = len(compressed_diff)
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}")

    # 解压缩并恢复目标张量
    decompress_time = time.time()
    print("Decompressing...")
    restored_tensor = decompress(
        base_tensor, compressed_diff, shape=tensor_size, dtype=np.float32
    )
    decompress_time = time.time() - decompress_time
    print(f"Decompression complete. Time is {decompress_time}")
    

    # 验证结果
    mse = torch.mean((target_tensor - restored_tensor) ** 2).item()
    print(f"Mean Squared Error (MSE) between original and restored tensors: {mse}")

    # 检查是否接近
    print("Original Tensor:")
    print(target_tensor)
    print("Restored Tensor:")
    print(restored_tensor)