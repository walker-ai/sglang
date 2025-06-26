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

void quant_per_block_int8_cuda(
            torch::Tensor input,
            torch::Tensor output,
            torch::Tensor scale,
            float sm_scale,
            int block_size,
            int tensor_layout);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From sage-attention
   */
  m.def(
      "quant_per_block_int8_cuda(Tensor input,"
                                "Tensor output,"
                                "Tensor scale,"
                                "float sm_scale,"
                                "int block_size,"
                                "int tensor_layout) -> ()");
  m.impl("quant_per_block_int8_cuda", torch::kCUDA, make_pytorch_shim(&quant_per_block_int8_cuda));
}

REGISTER_EXTENSION(sage_ops)
