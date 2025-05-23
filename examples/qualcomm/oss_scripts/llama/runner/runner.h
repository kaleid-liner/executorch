/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/io_manager.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

class Runner : public executorch::extension::llm::IRunner {
 public:
  explicit Runner(
      const std::vector<std::string>& models_path,
      const std::string& tokenizer_path,
      const std::string& performance_output_path_,
      const float logits_scale,
      const int32_t logits_offset,
      const float temperature,
      const int eval_mode,
      const std::string& kv_updater,
      const int num_iters);

  struct Stats {
    // Scaling factor for timestamps - in this case, we use ms.
    const long SCALING_FACTOR_UNITS_PER_SECOND = 1000;
    // Time stamps for the different stages of the execution
    // model_load_start_ms: Start of model loading.
    long model_load_start_ms;
    // model_load_end_ms: End of model loading.
    long model_load_end_ms;
    // inference_start_ms: Immediately after the model is loaded (or we check
    // for model load), measure the inference time.
    long inference_start_ms;
    // prompt_eval_end_ms: Prompt array allocation and tokenization. Ends right
    // before the inference loop starts
    long prompt_eval_end_ms;
    // first_token: Timestamp when the first generated token is emitted
    long first_token_ms;
    // inference_end_ms: End of inference/generation.
    long inference_end_ms;
    // Keep a running total of the time spent in sampling.
    long aggregate_sampling_time_ms;
    // Token count from prompt
    int64_t num_prompt_tokens;
    // Token count from generated (total - prompt)
    int64_t num_generated_tokens;
  };

  bool is_loaded() const override;
  executorch::runtime::Error load() override;
  executorch::runtime::Error generate(
      int32_t seq_len,
      const std::string& prompt,
      const std::string& system_prompt,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});
  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::extension::llm::Stats&)> stats_callback = {});
  void stop() override;
  std::vector<executorch::runtime::Result<executorch::runtime::MethodMeta>>
  get_methods_meta(std::string& method_name);

 private:
  enum LlamaVersion {
    kLlama2 = 0,
    kLlama3,
  };
  template <typename T>
  T getMetadataHelper(std::string method_name, T default_val);
  int32_t logitsToToken(
      const executorch::aten::Tensor& logits_tensor,
      int64_t pos);
  void run_model_step(
      const std::string& method_name,
      std::vector<std::vector<executorch::runtime::EValue>>& inputs);
  std::string prompt_;

  // metadata
  int32_t context_len_{0};
  int32_t prefill_ar_len_{0};
  int32_t prefill_cache_len_{0};
  int32_t kv_ar_len_{0};
  int32_t kv_cache_len_{0};
  int32_t vocab_size_;
  int32_t bos_id_;
  std::unordered_set<uint64_t> eos_id_;
  const int32_t n_bos_;
  const int32_t n_eos_;
  std::vector<std::shared_ptr<executorch::extension::Module>> modules_;
  std::string tokenizer_path_;
  std::string performance_output_path_;
  float logits_scale_;
  int32_t logits_offset_;
  float temperature_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<executorch::extension::llm::Sampler> sampler_;
  Stats stats_;
  std::unique_ptr<IoMgrBase> io_mgr_;
  EvalMode eval_mode_;
  bool use_int64_token_{false};
  std::string prefill_forward_name_;
  std::string kv_forward_name_;
  std::vector<std::string> method_names_;
  LlamaVersion llama_version_;
  std::string kv_updater_;
  int num_iters_;
  std::string kv_type_;
};

} // namespace example
