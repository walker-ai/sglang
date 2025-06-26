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
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_sage_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From sage-attention
   */

  // from fused

  using QuantFuncV1 = void(*)(torch::Tensor, torch::Tensor, torch::Tensor, float, int, int);
  using QuantFuncV2 = void(*)(torch::Tensor, torch::Tensor, torch::Tensor, int, int);
  m.def(
      "quant_per_block_int8_cuda(Tensor input,"
                                "Tensor output,"
                                "Tensor scale,"
                                "float sm_scale,"
                                "int block_size,"
                                "int tensor_layout) -> ()");
  m.impl("quant_per_block_int8_cuda", torch::kCUDA, make_pytorch_shim(static_cast<QuantFuncV1>(&quant_per_block_int8_cuda)));

  m.def(
      "quant_per_block_int8_cuda(Tensor input,"
                                "Tensor output,"
                                "Tensor scale,"
                                "int block_size,"
                                "int tensor_layout) -> ()");
  m.impl("quant_per_block_int8_cuda", torch::kCUDA, make_pytorch_shim(static_cast<QuantFuncV2>(&quant_per_block_int8_cuda)));

  m.def(
      "quant_per_block_int8_fuse_sub_mean_cuda(Tensor input,"
                                              "Tensor output,"
                                              "Tensor mean,"
                                              "Tensor output,"
                                              "Tensor scale,"
                                              "int block_size,"
                                              "int tensor_layout) -> ()");
  m.impl("quant_per_block_int8_fuse_sub_mean_cuda", torch::kCUDA, make_pytorch_shim(&quant_per_block_int8_fuse_sub_mean_cuda));

  m.def(
      "quant_per_warp_int8_cuda(Tensor input,"
                               "Tensor output,"
                               "Tensor scale,"
                               "int block_size,"
                               "int warp_blocK_size,"
                               "int tensor_layout) -> ()");
  m.impl("quant_per_warp_int8_cuda", torch::kCUDA, make_pytorch_shim(&quant_per_warp_int8_cuda));

  m.def(
      "sub_mean_cuda(Tensor input,"
                    "Tensor mean,"
                    "Tensor output,"
                    "int tensor_layout) -> ()");
  m.impl("sub_mean_cuda", torch::kCUDA, make_pytorch_shim(&sub_mean_cuda));

  m.def(
      "transpose_pad_permute_cuda(Tensor input,"
                                 "Tensor output,"
                                 "int tensor_layout) -> ()");
  m.impl("transpose_pad_permute_cuda", torch::kCUDA, make_pytorch_shim(&transpose_pad_permute_cuda));

  m.def(
      "scale_fuse_quant_cuda(Tensor input,"
                            "Tensor output,"
                            "Tensor scale,"
                            "int num_tokens,"
                            "float scale_max,"
                            "int tensor_layout) -> ()");
  m.impl("scale_fuse_quant_cuda", torch::kCUDA, make_pytorch_shim(&scale_fuse_quant_cuda));

  m.def(
      "mean_scale_fuse_quant_cuda(Tensor input,"
                                 "Tensor output,"
                                 "Tensor mean,"
                                 "Tensor scale,"
                                 "int num_tokens,"
                                 "float scale_max,"
                                 "int tensor_layout) -> ()");
  m.impl("mean_scale_fuse_quant_cuda", torch::kCUDA, make_pytorch_shim(&mean_scale_fuse_quant_cuda));
  
  // from qattn
  m.def(
      "qk_int8_sv_f8_accum_f32_attn_inst_buf(Tensor query,"
                                            "Tensor key,"
                                            "Tensor value,"
                                            "Tensor output,"
                                            "Tensor query_scale,"
                                            "Tensor key_scale,"
                                            "int tensor_layout,"
                                            "int is_causal,"
                                            "int qk_quant_gran,"
                                            "float sm_scale,"
                                            "int return_lse) -> Tensor");
  m.impl("qk_int8_sv_f8_accum_f32_attn_inst_buf", torch::kCUDA, make_pytorch_shim(&qk_int8_sv_f8_accum_f32_attn_inst_buf));

  m.def(
      "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(Tensor query,"
                                                         "Tensor key,"
                                                         "Tensor value,"
                                                         "Tensor output,"
                                                         "Tensor query_scale,"
                                                         "Tensor key_scale,"
                                                         "Tensor value_scale,"
                                                         "int tensor_layout,"
                                                         "int is_causal,"
                                                         "int qk_quant_gran,"
                                                         "float sm_scale,"
                                                         "int return_lse) -> Tensor");
  m.impl("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", torch::kCUDA, make_pytorch_shim(&qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf));
}

REGISTER_EXTENSION(sage_ops)