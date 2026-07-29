// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SGL_KERNEL_NPU_OPS_H
#define SGL_KERNEL_NPU_OPS_H

namespace sglang {
namespace npu_kernel {
at::Tensor helloworld(const at::Tensor &x, const at::Tensor &y);

at::Tensor cache_loc_assign(const at::Tensor &req_indices,
                            const at::Tensor &token_pool,
                            const at::Tensor &start_offset,
                            const at::Tensor &end_offset,
                            const at::Tensor &out_cache_loc);

at::Tensor cache_loc_update(const at::Tensor &req_indices,
                            const at::Tensor &token_pool,
                            const at::Tensor &start_offset,
                            const at::Tensor &end_offset,
                            const at::Tensor &out_cache_loc);

bool assign_cache_op(at::Tensor &dst_tensor, const at::Tensor &src_tensor,
                     const at::Tensor &dst_start_idx,
                     const at::Tensor &dst_end_idx,
                     const at::Tensor &src_start_idx,
                     const at::Tensor &src_end_idx);

void alloc_extend(const at::Tensor &pre_lens, const at::Tensor &seq_lens,
                  const at::Tensor &last_loc, const at::Tensor &free_pages,
                  int64_t pages_size, at::Tensor &out_indices,
                  at::Tensor &values);

void build_tree_efficient(
    const at::Tensor &parent_list, const at::Tensor &selected_index,
    const at::Tensor &verified_seq_len, const at::Tensor &tree_mask,
    const at::Tensor &positions, const at::Tensor &retrive_index,
    const at::Tensor &retrive_next_token,
    const at::Tensor &retrive_next_sibling, int64_t topk, int64_t depth,
    int64_t draft_token_num, int64_t tree_mask_mode);

void transfer_kv_dim_exchange(at::Tensor &device_k, at::Tensor &host_k,
                              at::Tensor &device_v, at::Tensor &host_v,
                              const at::Tensor &device_indices,
                              const at::Tensor &host_indices, int64_t page_size,
                              int64_t direction, int64_t flags);

at::Tensor bgmv_expand(at::Tensor &x, at::Tensor &weight, at::Tensor &indices,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size);

void bgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &indices,
                 at::Tensor &y, double scale);

at::Tensor sgmv_expand(at::Tensor &x, at::Tensor &weight,
                       at::Tensor &lora_indices, at::Tensor &seq_len,
                       at::Tensor &y, int64_t slice_offset, int64_t slice_size);

void sgmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices,
                 at::Tensor &seq_len, at::Tensor &y, double scale);

at::Tensor sgemmv_expand(at::Tensor &x, at::Tensor &weight,
                         at::Tensor &lora_indices, at::Tensor &seq_len,
                         at::Tensor &lora_ranks, at::Tensor &slice_offsets,
                         at::Tensor &y);

void sgemmv_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices,
                   at::Tensor &seq_len, at::Tensor &lora_ranks,
                   at::Tensor &lora_scales, at::Tensor &y);

at::Tensor sgemmc_expand(at::Tensor &x, at::Tensor &weight,
                         at::Tensor &lora_indices, at::Tensor &seq_len,
                         at::Tensor &lora_ranks, at::Tensor &slice_offsets,
                         at::Tensor &y);

void sgemmc_shrink(at::Tensor &x, at::Tensor &weight, at::Tensor &lora_indices,
                   at::Tensor &seq_len, at::Tensor &lora_ranks,
                   at::Tensor &lora_scales, at::Tensor &y, int64_t slice_count);

#ifdef BUILD_CATLASS_MODULE
void catlass_matmul_basic(const at::Tensor &tensor_a,
                          const at::Tensor &tensor_b, at::Tensor &tensor_c,
                          c10::optional<c10::string_view> format_mode);
#endif

at::Tensor apply_token_bitmask(at::Tensor logits, at::Tensor bitmask,
                               c10::optional<at::Tensor> indices);

} // namespace npu_kernel

} // namespace sglang

#endif // SGL_KERNEL_NPU_OPS_H
