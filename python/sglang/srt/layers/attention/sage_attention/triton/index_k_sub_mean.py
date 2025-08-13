import torch
import triton
import triton.language as tl

@triton.jit
def mean_normalize_kernel_k(
    k_ptr,                # [N, M, K]
    indices_ptr,          # [num_indices]
    output_ptr,           # [1, M, K]
    indices_selected_output_ptr, # [num_indices, M, K] - 选中的行
    cu_seqlens_k,
    mean_divisor,
    bs,
    N, M, K,              # 维度
    num_indices,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 获取当前线程块处理的M和K位置
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    if pid_b >= bs:
        return

    # 根据 pid_b 获取当前批次的有效索引范围
    cu_seqlens_k_start = tl.load(cu_seqlens_k + pid_b)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + pid_b + 1)
    
    # 获取当前批次的有效序列长度
    current_seq_len = cu_seqlens_k_end - cu_seqlens_k_start

    # 计算当前线程块处理的M和K的范围
    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K
    
    # 创建偏移量
    offset_m = tl.arange(0, BLOCK_SIZE_M)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算实际的M和K索引
    m_indices = m_start + offset_m
    k_indices = k_start + offset_k
    
    # 创建mask确保不越界
    m_mask = m_indices < M
    k_mask = k_indices < K
    mask = m_mask[:, None] & k_mask[None, :]
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # 遍历所有指定的索引，累加对应的值
    for i in range(current_seq_len):
        # 这里的 i 对应于当前批次中的相对位置
        # 我们需要找到它在整个 k cache 中的绝对位置
        global_idx = cu_seqlens_k_start + i

        # 加载索引
        idx = tl.load(indices_ptr + global_idx)
        
        # 计算当前索引对应的数据地址
        # k[idx, m_indices, k_indices]
        ptr = k_ptr + idx * M * K + m_indices[:, None] * K + k_indices[None, :]
        
        # 加载数据并累加
        data = tl.load(ptr, mask=mask, other=0.0)
        accumulator += data

        # 同时将选中的数据存储到 selected_output
        # selected_output[i, m_indices, k_indices] = data
        selected_ptr = indices_selected_output_ptr + global_idx * M * K + m_indices[:, None] * K + k_indices[None, :]
        tl.store(selected_ptr, data, mask=mask)
    

    # 计算输出地址并存储结果
    # output[0, m_indices, k_indices] = mean_result
    output_ptr_tile = output_ptr + m_indices[:, None] * K + k_indices[None, :]
    tl.atomic_add(output_ptr_tile, accumulator, mask=mask)

    divisor = tl.load(mean_divisor)

    # 计算均值 km
    final_sum = tl.load(output_ptr_tile, mask=mask, other=0.0)
    km = final_sum / divisor

    # 将最终结果存回 output
    tl.store(output_ptr_tile, km, mask=mask)

    # 遍历 indices_selected_output_ptr，执行减法
    for i in range(current_seq_len):
        global_idx = cu_seqlens_k_start + i
        
        # 加载之前存好的数据
        selected_ptr = indices_selected_output_ptr + global_idx * M * K + m_indices[:, None] * K + k_indices[None, :]
        data = tl.load(selected_ptr, mask=mask, other=0.0)
        
        # 执行减法
        normalized_data = data - km
        
        # 将结果写回 selected_ptr
        tl.store(selected_ptr, normalized_data, mask=mask)

@triton.jit
def mean_normalize_kernel_v(
    k_ptr,                # [N, M, K]
    indices_ptr,          # [num_indices]
    indices_selected_output_ptr, # [num_indices, M, K] - 选中的行
    cu_seqlens_v,
    bs,
    N, M, K,              # 维度
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 获取当前线程块处理的M和K位置
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_b = tl.program_id(axis=2)

    if pid_b >= bs:
        return

    # 根据 pid_b 获取当前批次的有效索引范围
    cu_seqlens_v_start = tl.load(cu_seqlens_v + pid_b)
    cu_seqlens_v_end = tl.load(cu_seqlens_v + pid_b + 1)
    
    # 获取当前批次的有效序列长度
    current_seq_len = cu_seqlens_v_end - cu_seqlens_v_start

    # 计算当前线程块处理的M和K的范围
    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K
    
    # 创建偏移量
    offset_m = tl.arange(0, BLOCK_SIZE_M)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算实际的M和K索引
    m_indices = m_start + offset_m
    k_indices = k_start + offset_k
    
    # 创建mask确保不越界
    m_mask = m_indices < M
    k_mask = k_indices < K
    mask = m_mask[:, None] & k_mask[None, :]
    

    # 遍历所有指定的索引，累加对应的值
    for i in range(current_seq_len):
        # 这里的 i 对应于当前批次中的相对位置
        # 我们需要找到它在整个 k cache 中的绝对位置
        global_idx = cu_seqlens_v_start + i

        # 加载索引
        idx = tl.load(indices_ptr + global_idx)
        
        # 计算当前索引对应的数据地址
        # k[idx, m_indices, k_indices]
        ptr = k_ptr + idx * M * K + m_indices[:, None] * K + k_indices[None, :]
        
        # 加载数据
        data = tl.load(ptr, mask=mask, other=0.0)

        # 同时将选中的数据存储到 selected_output
        # selected_output[i, m_indices, k_indices] = data
        selected_ptr = indices_selected_output_ptr + global_idx * M * K + m_indices[:, None] * K + k_indices[None, :]

        data_fp16 = data.to(tl.float16)

        tl.store(selected_ptr, data_fp16, mask=mask)

def triton_mean_normalize_k(k: torch.Tensor, indices: torch.Tensor, cu_seqlens_k: torch.Tensor):
    """
    使用 Triton 内核对指定索引的行在第0维上求均值。
    
    功能等价于：
    k_selected = k[indices]  # [num_indices, M, K]
    result = k_selected.mean(dim=0, keepdim=True)  # [1, M, K]
    
    :param k: 形状为 [N, M, K] 的张量
    :param indices: 形状为 [num_indices] 的索引张量
    :return: 形状为 [1, M, K] 的均值张量
    """
    
    # 确保输入在 GPU 上
    assert k.is_cuda and indices.is_cuda
    
    # 获取维度
    N, M, K = k.shape
    num_indices = indices.shape[0]
    
    # 创建输出张量
    output = torch.empty((1, M, K), device=k.device, dtype=torch.float32)
    indices_selected_output = torch.empty((num_indices, M, K), device=k.device, dtype=k.dtype)

    # 设置块大小
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_K = 128
    
    # 计算网格大小
    def cdiv(a, b):
        return (a + b - 1) // b
    
    grid_m = cdiv(M, BLOCK_SIZE_M)
    grid_k = cdiv(K, BLOCK_SIZE_K)

    # 启动内核
    mean_normalize_kernel_k[(grid_m, grid_k)](
        k_ptr=k,
        indices_ptr=indices,
        output_ptr=output,
        indices_selected_output_ptr=indices_selected_output,
        cu_seqlens_k=cu_seqlens_k,
        mean_divisor=cu_seqlens_k[-1],
        bs=len(cu_seqlens_k) - 1,
        N=N, M=M, K=K,
        num_indices=num_indices,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return indices_selected_output

def triton_mean_normalize_v(v: torch.Tensor, indices: torch.Tensor, cu_seqlens_v: torch.Tensor):
    """
    使用 Triton 内核对指定索引的行在第0维上求均值。
    
    功能等价于：
    v_selected = v[indices]  # [num_indices, M, K]
    result = v_selected.mean(dim=0, keepdim=True)  # [1, M, K]
    
    :param v: 形状为 [N, M, K] 的张量
    :param indices: 形状为 [num_indices] 的索引张量
    :return: 形状为 [1, M, K] 的均值张量
    """
    
    # 确保输入在 GPU 上
    assert v.is_cuda and indices.is_cuda
    
    # 获取维度
    N, M, K = v.shape
    num_indices = indices.shape[0]
    
    # 创建输出张量
    indices_selected_output = torch.empty((num_indices, M, K), device=v.device, dtype=torch.float16)

    # 设置块大小
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_K = 128

    # 计算网格大小
    def cdiv(a, b):
        return (a + b - 1) // b

    grid_m = cdiv(M, BLOCK_SIZE_M)
    grid_k = cdiv(K, BLOCK_SIZE_K)

    # 启动内核
    mean_normalize_kernel_v[(grid_m, grid_k)](
        k_ptr=v,
        indices_ptr=indices,
        indices_selected_output_ptr=indices_selected_output,
        cu_seqlens_v=cu_seqlens_v,
        bs=len(cu_seqlens_v) - 1,
        N=N, M=M, K=K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    return indices_selected_output

# ----------------------------
# Kernel 1: 对每个 selected index 做加载 -> 写入 selected_output -> atomic_add 到 output
# 每个 kernel block 负责 (BLOCK_SIZE_M x BLOCK_SIZE_K) 的 tile，
# grid = (grid_m, grid_k, num_indices)
# ----------------------------
@triton.jit
def kernel_accumulate_one_index_k(
    k_ptr,                      # pointer to k: shape [N, M, K]
    indices_ptr,                # pointer to indices: shape [num_indices]
    output_ptr,                 # pointer to output: shape [1, M, K] (float32)
    indices_selected_output_ptr,# pointer to selected output: shape [num_indices, M, K]
    M, K, num_indices,
    actual_indice,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_INDICE: tl.constexpr,    # 新增编译期常量
):
    pid_m = tl.program_id(axis=0)   # tile index along M
    pid_k = tl.program_id(axis=1)   # tile index along K
    pid_idx = tl.program_id(axis=2) # which selected index (0..num_blocks-1)

    offs_idx = tl.arange(0, BLOCK_SIZE_INDICE)   # 新增，用来向量化处理多个索引

    # 计算实际索引
    idxs = pid_idx * BLOCK_SIZE_INDICE + offs_idx

    # 越界mask，防止访问超过num_indices
    valid_mask = idxs < tl.load(actual_indice)

    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_idx = m_start + offs_m            # (BLOCK_SIZE_M,)
    k_idx = k_start + offs_k            # (BLOCK_SIZE_K,)

    # masks for bounds
    m_mask = m_idx < M
    k_mask = k_idx < K
    mask = m_mask[:, None] & k_mask[None, :]

    # load the actual global idx from indices, shape (INDICES_PER_BLOCK,)
    idx_vals = tl.load(indices_ptr + idxs, mask=valid_mask, other=0)   # shape (INDICES_PER_BLOCK,)

    # base pointers (elements)
    # k layout assumed contiguous as [N, M, K] -> index calc: idx * (M*K) + m * K + k
    base_ks = idx_vals * (M * K)

    # ptrs shape will be broadcasted to (INDICES_PER_BLOCK, BLOCK_SIZE_M, BLOCK_SIZE_K)
    ptrs = base_ks[:, None, None] + m_idx[None, :, None] * K + k_idx[None, None, :]

    # load data from k
    load_mask = valid_mask[:, None, None] & mask[None, :, :]
    data = tl.load(k_ptr + ptrs, mask=load_mask, other=0.0)  # dtype = k.dtype (e.g., float16)

    # write to indices_selected_output[pid_idx * INDICES_PER_BLOCK + offs_idx, m_idx, k_idx]
    base_selected = idxs * (M * K)
    selected_ptrs = base_selected[:, None, None] + m_idx[None, :, None] * K + k_idx[None, None, :]
    tl.store(indices_selected_output_ptr + selected_ptrs, data, mask=load_mask)

    # atomic add to output[0, m_idx, k_idx]
    base_output = 0  # since output is [1, M, K] with leading dim 1
    out_ptrs = base_output + m_idx[:, None] * K + k_idx[None, :]
    # ensure we're adding in float32 to preserve accumulation precision (like original)
    # Triton atomic_add supports same dtype; if k is float16 we cast
    # We'll cast loaded data to float32 for atomic add (typical practice).

    data_fp32 = tl.cast(data, tl.float32)

    # 对所有有效indices求和：tl.sum(axis=0)
    acc = tl.sum(data_fp32, axis=0)

    tl.atomic_add(output_ptr + out_ptrs, acc, mask=mask)

# ----------------------------
# Kernel 1: 对每个 selected index 做加载 -> 写入 selected_output -> atomic_add 到 output
# 每个 kernel block 负责 (BLOCK_SIZE_M x BLOCK_SIZE_K) 的 tile，
# grid = (grid_m, grid_k, num_indices)
# ----------------------------
@triton.jit
def kernel_accumulate_one_index_v(
    v_ptr,                      # pointer to k: shape [N, M, K]
    indices_ptr,                # pointer to indices: shape [num_indices]
    indices_selected_output_ptr,# pointer to selected output: shape [num_indices, M, K]
    N, M, K, num_indices,
    actual_indice,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_INDICE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)   # tile index along M
    pid_k = tl.program_id(axis=1)   # tile index along K
    pid_idx = tl.program_id(axis=2) # which selected index (0..num_indices-1)

    offs_idx = tl.arange(0, BLOCK_SIZE_INDICE)   # 新增，用来向量化处理多个索引

    # 计算实际索引
    idxs = pid_idx * BLOCK_SIZE_INDICE + offs_idx


    # 越界mask，防止访问超过num_indices
    valid_mask = idxs < tl.load(actual_indice)

    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_idx = m_start + offs_m            # (BLOCK_SIZE_M,)
    k_idx = k_start + offs_k            # (BLOCK_SIZE_K,)

    # masks for bounds
    m_mask = m_idx < M
    k_mask = k_idx < K
    mask = m_mask[:, None] & k_mask[None, :]

    # load the actual global idx from indices, shape (INDICES_PER_BLOCK,)
    idx_vals = tl.load(indices_ptr + idxs, mask=valid_mask, other=0)   # shape (INDICES_PER_BLOCK,)

    # base pointers (elements)
    # k layout assumed contiguous as [N, M, K] -> index calc: idx * (M*K) + m * K + k
    base_ks = idx_vals * (M * K)

    # ptrs shape will be broadcasted to (INDICES_PER_BLOCK, BLOCK_SIZE_M, BLOCK_SIZE_K)
    ptrs = base_ks[:, None, None] + m_idx[None, :, None] * K + k_idx[None, None, :]

    # load data from v
    load_mask = valid_mask[:, None, None] & mask[None, :, :]
    data = tl.load(v_ptr + ptrs, mask=load_mask, other=0.0)  # dtype = v.dtype (e.g., float16)

    # write to indices_selected_output[pid_idx * INDICES_PER_BLOCK + offs_idx, m_idx, k_idx]
    base_selected = idxs * (M * K)
    selected_ptrs = base_selected[:, None, None] + m_idx[None, :, None] * K + k_idx[None, None, :]

    data_fp16 = tl.cast(data, tl.float16)
    tl.store(indices_selected_output_ptr + selected_ptrs, data_fp16, mask=load_mask)

# ----------------------------
# Kernel 2: 将 output 除以 mean_divisor（标量）
# grid = (grid_m, grid_k)
# ----------------------------
@triton.jit
def kernel_divide_output_by_divisor(
    output_ptr,      # float32 [1, M, K]
    mean_divisor,    # float32 scalar (passed as a tensor element)
    M, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_idx = m_start + offs_m
    k_idx = k_start + offs_k

    m_mask = m_idx < M
    k_mask = k_idx < K
    mask = m_mask[:, None] & k_mask[None, :]

    base_output = 0
    out_ptrs = base_output + m_idx[:, None] * K + k_idx[None, :]

    out_tile = tl.load(output_ptr + out_ptrs, mask=mask, other=0.0)  # float32
    # mean_divisor is a pointer to a scalar tensor element; we receive it as python arg, so pass in as scalar
    mean_val = tl.load(mean_divisor)
    # Avoid division by zero (not expected if mean_divisor>0)
    res = out_tile / mean_val
    tl.store(output_ptr + out_ptrs, res, mask=mask)


# ----------------------------
# Kernel 3: 每个 selected-index 的 tile 减去 mean（output 已经被除以 divisor）
# grid = (grid_m, grid_k, num_indices)
# ----------------------------
@triton.jit
def kernel_subtract_mean_from_selected(
    output_ptr,                 # float32 [1, M, K] (mean already computed)
    indices_selected_output_ptr,# dtype same as k
    actual_indice,
    num_indices,
    M, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_INDICE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_idx = tl.program_id(axis=2)

    offs_idx = tl.arange(0, BLOCK_SIZE_INDICE)

    idxs = pid_idx * BLOCK_SIZE_INDICE + offs_idx  # 计算真实索引范围

    valid_mask = idxs < tl.load(actual_indice)  # 越界掩码

    m_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    m_idx = m_start + offs_m
    k_idx = k_start + offs_k

    m_mask = m_idx < M
    k_mask = k_idx < K
    mask = m_mask[:, None] & k_mask[None, :]

    # load selected data
    base_selected = idxs * (M * K)  # shape (INDICES_PER_BLOCK)
    selected_ptrs = base_selected[:, None, None] + m_idx[None, :, None] * K + k_idx[None, None, :]
    load_mask = valid_mask[:, None, None] & mask[None, :, :]
    sel = tl.load(indices_selected_output_ptr + selected_ptrs, mask=load_mask, other=0.0)

    # load mean (output)
    base_output = 0
    out_ptrs = base_output + m_idx[:, None] * K + k_idx[None, :]
    mean_tile = tl.load(output_ptr + out_ptrs, mask=mask, other=0.0)  # float32

    # cast mean to selected dtype if needed
    mean_cast = tl.cast(mean_tile, tl.bfloat16)
    normalized = sel - mean_cast
    tl.store(indices_selected_output_ptr + selected_ptrs, normalized, mask=load_mask)


# ----------------------------
# Host wrapper: triton_mean_normalize_k
# ----------------------------
def triton_mean_normalize_k_gpt(k: torch.Tensor, indices: torch.Tensor, cu_seqlens_k: torch.Tensor, mean_k: torch.Tensor, indices_selected_k_output: torch.Tensor):
    """
    k: [N, M, K] (any dtype)
    indices: [num_indices] int32/64
    cu_seqlens_k: prefix sums for batches, shape [bs + 1]
                  NOTE: this function preserves the original "mean_divisor" semantics used in your snippet:
                  mean_divisor = cu_seqlens_k[-1]  (a scalar)
    returns: indices_selected_output tensor, shape [num_indices, M, K], dtype same as k
    """
    assert k.is_cuda and indices.is_cuda and cu_seqlens_k.is_cuda

    N, M, K = k.shape
    num_indices = indices.shape[0]

    # output as float32 accumulation (same as original)
    # output = torch.zeros((1, M, K), device=k.device, dtype=torch.float32)
    output = mean_k
    # indices_selected_output = torch.zeros((130, M, K), device=k.device, dtype=k.dtype)
    indices_selected_output = indices_selected_k_output

    # tile sizes (kept same as you used)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_INDICE = 8

    def cdiv(a, b):
        return (a + b - 1) // b

    grid_m = cdiv(M, BLOCK_SIZE_M)
    grid_k = cdiv(K, BLOCK_SIZE_K)
    grid_indice = cdiv(4096, BLOCK_SIZE_INDICE)

    # Kernel 1: accumulate per selected index
    # Note: grid is (grid_m, grid_k, num_indices)

    # 暂时用DTYPE_ID代替，0代表k的输出 dtype = tl.float32
    grid1 = (grid_m, grid_k, grid_indice)
    kernel_accumulate_one_index_k[grid1](
        k, indices, output, indices_selected_output,
        M, K, num_indices,
        cu_seqlens_k[-1],
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_INDICE=BLOCK_SIZE_INDICE,
    )

    # Kernel 2: divide output by mean_divisor (use same mean_divisor as your snippet)
    # Your original snippet passed mean_divisor = cu_seqlens_k[-1]
    # We will read that scalar on host and pass it in
    grid2 = (grid_m, grid_k)
    kernel_divide_output_by_divisor[grid2](
        output, 
        cu_seqlens_k[-1],
        M, 
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

    # Kernel 3: subtract mean from selected outputs
    grid3 = (grid_m, grid_k, grid_indice)
    kernel_subtract_mean_from_selected[grid3](
        output, indices_selected_output, 
        cu_seqlens_k[-1],
        num_indices, M, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_INDICE=BLOCK_SIZE_INDICE,
    )

    return indices_selected_output, cu_seqlens_k[-1]

def triton_mean_normalize_v_gpt(v: torch.Tensor, indices: torch.Tensor, cu_seqlens_v: torch.Tensor, indices_selected_v_output: torch.Tensor):
    """
    v: [N, M, K] (any dtype)
    indices: [num_indices] int32/64
    cu_seqlens_v: prefix sums for batches, shape [bs + 1]
                  NOTE: this function preserves the original "mean_divisor" semantics used in your snippet:
                  mean_divisor = cu_seqlens_k[-1]  (a scalar)
    returns: indices_selected_output tensor, shape [num_indices, M, K], dtype same as k
    """
    assert v.is_cuda and indices.is_cuda and cu_seqlens_v.is_cuda

    N, M, K = v.shape
    num_indices = indices.shape[0]

    # indices_selected_output = torch.zeros((130, M, K), device=v.device, dtype=torch.float16)
    indices_selected_output = indices_selected_v_output

    # tile sizes (kept same as you used)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_INDICE = 8

    bs = cu_seqlens_v[-1] - 1

    def cdiv(a, b):
        return (a + b - 1) // b

    grid_m = cdiv(M, BLOCK_SIZE_M)
    grid_k = cdiv(K, BLOCK_SIZE_K)
    grid_indice = cdiv(4096, BLOCK_SIZE_INDICE)


    grid1 = (grid_m, grid_k, grid_indice)
    kernel_accumulate_one_index_v[grid1](
        v, indices, indices_selected_output,
        N, M, K, num_indices,
        cu_seqlens_v[-1],
        BLOCK_SIZE_M=BLOCK_SIZE_M, 
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_INDICE=BLOCK_SIZE_INDICE,
    )

    return indices_selected_output

# -----------------------------------------------------------------------------
# 示例用法和验证
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 在 GPU 上创建模拟数据
    N, M, K = 551276, 8, 128
    num_indices = 1000
    
    # 创建测试数据
    k_tensor = torch.randn(N, M, K, device='cuda')  # 使用随机数便于验证
    indices_tensor = torch.randint(0, N, (num_indices,), device='cuda')

    mean_k = torch.zeros((1, M, K), device='cuda', dtype=k_tensor.dtype)
    indices_selected_k_output = torch.zeros((num_indices, M, K), device='cuda', dtype=k_tensor.dtype)
    cu_seqlens_k = torch.tensor([0, num_indices], device='cuda')
    
    print(f"输入张量形状: {k_tensor.shape}")
    print(f"索引张量: {indices_tensor}")
    print(f"索引数量: {num_indices}")
    
    # 调用 Triton 内核
    result_triton, _ = triton_mean_normalize_k_gpt(k_tensor, indices_tensor, cu_seqlens_k, mean_k, indices_selected_k_output)
    
    # 使用 PyTorch 原生操作进行验证
    # 步骤1: 根据索引切分
    k_selected = k_tensor[indices_tensor]  # [num_indices, M, K]
    print(f"选中的张量形状: {k_selected.shape}")
    
    # 步骤2: 在第0维求均值
    result_torch = k_selected#k_selected.mean(dim=0, keepdim=True)  # [1, M, K]
    
    print(f"Triton结果形状: {result_triton.shape}")
    print(f"PyTorch结果形状: {result_torch.shape}")
    
    # 比较结果
    if torch.allclose(result_triton, result_torch, atol=1e-5):
        print("✅ Triton 内核运行成功！结果与PyTorch一致")
    else:
        print("❌ 结果不一致！")
        max_diff = torch.max(torch.abs(result_triton - result_torch)).item()
        print(f"最大差异: {max_diff}")
        
        # 打印一些调试信息
        print(f"Triton结果样本: {result_triton[0, 0, :5]}")
        print(f"PyTorch结果样本: {result_torch[0, 0, :5]}")
    
    # 验证功能逻辑
    print("\n=== 功能验证 ===")
    print("验证：从大张量中选择特定行，然后对这些行求均值")
    
    # 手动验证几个位置
    manual_sum = torch.zeros((M, K), device='cuda')
    for i, idx in enumerate(indices_tensor[:3]):  # 只检查前3个
        print(f"索引 {i}: 选择第 {idx.item()} 行")
        selected_row = k_tensor[idx]  # [M, K]
        manual_sum += selected_row
        print(f"  该行 [0,0] 位置的值: {selected_row[0, 0].item():.6f}")
    
    print(f"前3行在 [0,0] 位置的平均值: {(manual_sum[0, 0] / 3).item():.6f}")
    print(f"完整结果在 [0,0] 位置的值: {result_triton[0, 0, 0].item():.6f}")