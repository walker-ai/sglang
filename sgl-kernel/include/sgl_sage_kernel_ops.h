/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <Python.h>
#include <torch/library.h>
#include <torch/torch.h>

#include <vector>

#include "sgl_kernel_torch_shim.h"

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)

#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define REGISTER_EXTENSION(NAME)                                                                      \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                                            \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                                                  \
  }

/*
 * From sage-attention
 */

// from fused
void quant_per_block_int8_cuda_with_sm_scale(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_fuse_sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_warp_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int warp_block_size,
                int tensor_layout);

void sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                int tensor_layout);

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int tensor_layout);

void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

void mean_scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor mean,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

// from qattn
torch::Tensor qk_int8_sv_f8_accum_f32_attn_inst_buf(
                    torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);

torch::Tensor qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
                    torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    int tensor_layout,
                    int is_causal,
                    int qk_quant_gran,
                    float sm_scale,
                    int return_lse);