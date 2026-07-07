/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

void GemmGatedActSM90(TensorView workspace_buffer, TensorView a, TensorView weight,
                      Optional<TensorView> bias, Optional<TensorView> alpha_tensor,
                      Optional<TensorView> out_scale, TensorView out, double alpha,
                      int64_t sm_count, int64_t tactic);

// "Fused gated-activation GEMM (gated-MLP FC1) for SM90"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_gated_act_sm90, GemmGatedActSM90);
