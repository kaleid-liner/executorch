/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/executor/method_meta.h>

namespace example {

enum EvalMode {
  kKVCached = 0,
  kHybrid,
  kUnsupported,
};
class IoMgrBase {
 public:
  IoMgrBase(
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules);
  virtual ~IoMgrBase();
  virtual void init_io() = 0;
  virtual void reset_io(
      const std::vector<executorch::runtime::Result<
          executorch::runtime::MethodMeta>>& prefill_methods_meta,
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          kv_methods_meta) = 0;
  virtual void prepare_prefill_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) = 0;
  virtual void prepare_kv_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) = 0;
  virtual void fill_prefill_toks(
      int64_t start_pos,
      std::vector<uint64_t>& prompt_tokens) = 0;
  virtual void fill_kv_tok_mask(int64_t pos, int64_t cur_token) = 0;
  virtual void update_prefill_to_kv_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors) = 0;
  virtual void update_kv_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors) = 0;
  virtual void update_prefill_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors) = 0;
  void* get_mutable_ptr();
  std::vector<executorch::aten::Tensor> get_input_tensors(
      int shard_index,
      const std::string& method_name);
  std::vector<executorch::aten::Tensor> get_output_tensors(
      int shard_index,
      const std::string& method_name);

 protected:
  std::unique_ptr<void, void (*)(void*)> data_ptr_;
  std::unordered_map<
      std::string,
      std::vector<std::vector<executorch::aten::TensorImpl*>>>
      input_tensors_;
  std::unordered_map<
      std::string,
      std::vector<std::vector<executorch::aten::TensorImpl*>>>
      output_tensors_;
  std::vector<std::shared_ptr<executorch::extension::Module>> modules_;
};

class ShiftPointerIoMgr : public IoMgrBase {
 public:
  ShiftPointerIoMgr(
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules,
      int32_t context_len,
      int32_t prefill_ar_len,
      int32_t prefill_cache_len,
      int32_t kv_ar_len,
      int32_t kv_cache_len,
      int32_t vocab_size,
      int32_t num_layers,
      int32_t head_dim,
      int32_t num_heads,
      EvalMode eval_mode,
      const std::string& prefill_forward_name,
      const std::string& kv_forward_name,
      const bool use_int64_token);

  void init_io() override;
  void reset_io(
      const std::vector<executorch::runtime::Result<
          executorch::runtime::MethodMeta>>& prefill_methods_meta,
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          kv_methods_meta) override;
  void prepare_prefill_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) override;
  void prepare_kv_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) override;
  void fill_prefill_toks(
      int64_t start_pos,
      std::vector<uint64_t>& prompt_tokens) override;
  void fill_kv_tok_mask(int64_t pos, int64_t cur_token) override;
  void update_prefill_to_kv_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  void update_kv_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  void update_prefill_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  struct IO {
    int64_t kv_input_toks;
    int32_t kv_input_pos;
    std::vector<std::vector<std::vector<uint8_t>>> k_cache;
    std::vector<std::vector<uint8_t>> v_cache;
    std::vector<std::vector<uint8_t>> k_cache_out;
    std::vector<uint16_t> kv_attention_mask;
    std::vector<uint16_t> kv_logits;
    std::vector<int64_t> prefill_input_toks;
    std::vector<int32_t> prefill_input_pos;
    std::vector<uint16_t> prefill_attention_mask;
    std::vector<uint16_t> prefill_logits;
  };

 private:
  // If the cache length is zero, it indicates a BERT model, which does not use
  // position ids or KV cache inputs.
  bool is_bert() const {
    return prefill_cache_len_ == 0;
  }
  std::unique_ptr<executorch::aten::TensorImpl> kv_input_toks_;
  std::unique_ptr<executorch::aten::TensorImpl> kv_input_pos_;
  std::unique_ptr<executorch::aten::TensorImpl> kv_attention_mask_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_input_toks_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_input_pos_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_attention_mask_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_logits_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      k_cache_in_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      v_cache_in_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      k_cache_out_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      v_cache_out_;
  std::unique_ptr<executorch::aten::TensorImpl> kv_logits_;
  std::vector<int> shard_layers_;
  int32_t context_len_{0};
  int32_t kv_ar_len_{0};
  int32_t kv_cache_len_{0};
  int32_t prefill_ar_len_{0};
  int32_t prefill_cache_len_{0};
  int32_t vocab_size_;
  int32_t num_layers_;
  int32_t head_dim_;
  int32_t num_heads_;
  EvalMode eval_mode_;
  std::string prefill_forward_name_;
  std::string kv_forward_name_;
  const bool use_int64_token_{false};
};

template <typename KVType, typename IOType>
class SmartMaskIoMgr : public IoMgrBase {
 public:
  SmartMaskIoMgr(
      std::vector<std::shared_ptr<executorch::extension::Module>>& modules,
      int32_t context_len,
      int32_t prefill_ar_len,
      int32_t prefill_cache_len,
      int32_t kv_ar_len,
      int32_t kv_cache_len,
      int32_t vocab_size,
      int32_t num_layers,
      int32_t head_dim,
      int32_t num_heads,
      EvalMode eval_mode,
      const std::string& prefill_forward_name,
      const std::string& kv_forward_name,
      const bool use_int64_token);

  void init_io() override;
  void reset_io(
      const std::vector<executorch::runtime::Result<
          executorch::runtime::MethodMeta>>& prefill_methods_meta,
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          kv_methods_meta) override;
  void prepare_prefill_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) override;
  void prepare_kv_io(
      const std::vector<
          executorch::runtime::Result<executorch::runtime::MethodMeta>>&
          methods_meta) override;
  void fill_prefill_toks(
      int64_t start_pos,
      std::vector<uint64_t>& prompt_tokens) override;
  void fill_kv_tok_mask(int64_t pos, int64_t cur_token) override;
  void update_prefill_to_kv_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  void update_kv_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;
  void update_prefill_io(
      int64_t cur_token,
      int64_t pos,
      std::vector<std::vector<executorch::aten::Tensor>>& output_tensors)
      override;

  std::unordered_map<std::string, size_t> get_io_elements();
  std::unordered_map<std::string, size_t> get_io_bytes();

  struct IO {
    void* shared_buffer_base;
    int64_t* kv_input_toks;
    int32_t* kv_input_pos;
    // layer -> head -> head_dim * seq_len
    std::vector<std::vector<KVType*>> k_cache;
    std::vector<std::vector<KVType*>> v_cache;
    // layer -> head -> head_dim
    std::vector<std::vector<KVType*>> k_cache_out;
    std::vector<std::vector<KVType*>> v_cache_out;
    // kv_ar_len_ * context_len_
    IOType* kv_attention_mask;
    // kv_ar_len_ * vocab_size
    IOType* kv_logits;
    // prefill_ar_len_
    int64_t* prefill_input_toks;
    int32_t* prefill_input_pos;
    // prefill_ar_len_ * context_len_
    IOType* prefill_attention_mask;
    // vocab_size * prefill_ar_len_
    IOType* prefill_logits;

    size_t num_layers_;
    size_t num_heads_;
    size_t head_dim_;
    std::unordered_map<std::byte*, size_t> io_pos_map;
    ~IO() {
      QnnExecuTorchFreeCustomMem(shared_buffer_base);
    }
    void init_io_ptrs(
        void* shared_buffer_ptr,
        std::unordered_map<std::string, size_t>& io_bytes_map);
    void add_custom_mem_info(
        void* ptr,
        size_t nbytes,
        executorch::aten::ScalarType scalar_type,
        executorch::runtime::TensorInfo& tensor_info);
  };

 private:
  // If the cache length is zero, it indicates a BERT model, which does not use
  // position ids or KV cache inputs.
  bool is_bert() const {
    return prefill_cache_len_ == 0;
  }
  std::unique_ptr<executorch::aten::TensorImpl> kv_input_toks_;
  std::unique_ptr<executorch::aten::TensorImpl> kv_input_pos_;
  std::unique_ptr<executorch::aten::TensorImpl> kv_attention_mask_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_input_toks_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_input_pos_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_attention_mask_;
  std::unique_ptr<executorch::aten::TensorImpl> prefill_logits_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      k_cache_in_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      v_cache_in_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      k_cache_out_;
  std::unordered_map<
      std::string,
      std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      v_cache_out_;
  std::unique_ptr<executorch::aten::TensorImpl> kv_logits_;
  std::vector<int> shard_layers_;
  int32_t context_len_{0};
  int32_t kv_ar_len_{0};
  int32_t kv_cache_len_{0};
  int32_t prefill_ar_len_{0};
  int32_t prefill_cache_len_{0};
  int32_t vocab_size_;
  int32_t num_layers_;
  int32_t head_dim_;
  int32_t num_heads_;
  EvalMode eval_mode_;
  std::string prefill_forward_name_;
  std::string kv_forward_name_;
  const bool use_int64_token_{false};
};

} // namespace example
