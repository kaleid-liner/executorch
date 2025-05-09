/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <executorch/runtime/core/array_ref.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/result.h>
// Enable flag for test
#define ET_EVENT_TRACER_ENABLED
#include <executorch/runtime/core/event_tracer_hooks.h>
#include <executorch/runtime/core/event_tracer_hooks_delegate.h>

using executorch::aten::Tensor;
using executorch::runtime::AllocatorID;
using executorch::runtime::ArrayRef;
using executorch::runtime::ChainID;
using executorch::runtime::DebugHandle;
using executorch::runtime::DelegateDebugIntId;
using executorch::runtime::EValue;
using executorch::runtime::EventTracer;
using executorch::runtime::EventTracerDebugLogLevel;
using executorch::runtime::EventTracerEntry;
using executorch::runtime::EventTracerFilterBase;
using executorch::runtime::kUnsetChainId;
using executorch::runtime::kUnsetDebugHandle;
using executorch::runtime::kUnsetDelegateDebugIntId;
using executorch::runtime::LoggedEValueType;
using executorch::runtime::Result;

class DummyEventTracer : public EventTracer {
 public:
  DummyEventTracer() {}

  ~DummyEventTracer() override {}

  void create_event_block(const char* name) override {
    (void)name;
    return;
  }

  EventTracerEntry start_profiling(
      const char* name,
      ChainID chain_id = kUnsetChainId,
      DebugHandle debug_handle = kUnsetDebugHandle) override {
    (void)chain_id;
    (void)debug_handle;
    ET_CHECK(strlen(name) + 1 < sizeof(event_name_));
    memcpy(event_name_, name, strlen(name) + 1);
    return EventTracerEntry();
  }

  void end_profiling(EventTracerEntry prof_entry) override {
    (void)prof_entry;
    memset(event_name_, 0, sizeof(event_name_));
    return;
  }

  void track_allocation(AllocatorID id, size_t size) override {
    (void)id;
    (void)size;
    return;
  }

  AllocatorID track_allocator(const char* name) override {
    (void)name;
    return 0;
  }

  EventTracerEntry start_profiling_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_id) override {
    (void)name;
    (void)delegate_debug_id;
    return EventTracerEntry();
  }

  void end_profiling_delegate(
      ET_UNUSED EventTracerEntry event_tracer_entry,
      ET_UNUSED const void* metadata,
      ET_UNUSED size_t metadata_len) override {
    (void)event_tracer_entry;
    (void)metadata;
    (void)metadata_len;
  }

  void set_delegation_intermediate_output_filter(
      EventTracerFilterBase* event_tracer_filter) override {
    (void)event_tracer_filter;
  }

  void log_profiling_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_id,
      et_timestamp_t start_time,
      et_timestamp_t end_time,
      const void* metadata,
      size_t metadata_len = 0) override {
    (void)name;
    (void)delegate_debug_id;
    (void)start_time;
    (void)end_time;
    (void)metadata;
    (void)metadata_len;
  }

  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const Tensor& output) override {
    (void)name;
    (void)delegate_debug_index;
    (void)output;
    return true;
  }

  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const ArrayRef<Tensor> output) override {
    (void)name;
    (void)delegate_debug_index;
    (void)output;
    return true;
  }

  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const int& output) override {
    (void)name;
    (void)delegate_debug_index;
    (void)output;
    return true;
  }

  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const bool& output) override {
    (void)name;
    (void)delegate_debug_index;
    (void)output;
    return true;
  }

  virtual Result<bool> log_intermediate_output_delegate(
      const char* name,
      DelegateDebugIntId delegate_debug_index,
      const double& output) override {
    (void)name;
    (void)delegate_debug_index;
    (void)output;
    return true;
  }

  Result<bool> log_evalue(const EValue& evalue, LoggedEValueType evalue_type)
      override {
    logged_evalue_ = evalue;
    logged_evalue_type_ = evalue_type;
    return true;
  }

  EValue logged_evalue() {
    return logged_evalue_;
  }

  LoggedEValueType logged_evalue_type() {
    return logged_evalue_type_;
  }

  char* get_event_name() {
    return event_name_;
  }

  void reset_logged_value() {
    logged_evalue_ = EValue(false);
  }

 private:
  EValue logged_evalue_ = EValue(false);
  LoggedEValueType logged_evalue_type_;
  char event_name_[1024];
};

/**
 * Exercise all the event_tracer API's for a basic sanity check.
 */
void RunSimpleTracerTest(EventTracer* event_tracer) {
  using executorch::runtime::internal::event_tracer_begin_profiling_event;
  using executorch::runtime::internal::event_tracer_create_event_block;
  using executorch::runtime::internal::event_tracer_end_profiling_event;
  using executorch::runtime::internal::event_tracer_track_allocation;
  using executorch::runtime::internal::event_tracer_track_allocator;
  using executorch::runtime::internal::EventTracerProfileInstructionScope;
  using executorch::runtime::internal::EventTracerProfileMethodScope;

  event_tracer_create_event_block(event_tracer, "ExampleEvent");
  event_tracer_create_event_block(event_tracer, "ExampleEvent");
  EventTracerEntry event_entry =
      event_tracer_begin_profiling_event(event_tracer, "ExampleEvent");
  event_tracer_end_profiling_event(event_tracer, event_entry);
  {
    EventTracerProfileMethodScope event_tracer_profile_scope(
        event_tracer, "ExampleScope");
  }
  {
    EventTracerProfileInstructionScope event_tracer_profile_instruction_scope(
        event_tracer, 0, 1);
  }
  AllocatorID allocator_id =
      event_tracer_track_allocator(event_tracer, "AllocatorName");
  event_tracer_track_allocation(event_tracer, allocator_id, 64);
}

TEST(TestEventTracer, SimpleEventTracerTest) {
  // Call all the EventTracer macro's with a valid pointer to an event tracer
  // and also with a null pointer (to test that the null case works).
  DummyEventTracer dummy;
  std::vector<DummyEventTracer*> dummy_event_tracer_arr = {&dummy, nullptr};
  for (const auto i : c10::irange(dummy_event_tracer_arr.size())) {
    RunSimpleTracerTest(&dummy);
    RunSimpleTracerTest(nullptr);
  }
}

/**
 * Exercise all the event_tracer API's for delegates as a basic sanity check.
 */
void RunSimpleTracerTestDelegate(EventTracer* event_tracer) {
  EventTracerEntry event_tracer_entry = event_tracer_start_profiling_delegate(
      event_tracer, "test_event", kUnsetDelegateDebugIntId);
  event_tracer_end_profiling_delegate(
      event_tracer, event_tracer_entry, nullptr);
  event_tracer_start_profiling_delegate(event_tracer, nullptr, 1);
  event_tracer_end_profiling_delegate(
      event_tracer, event_tracer_entry, "test_metadata");
  event_tracer_log_profiling_delegate(
      event_tracer, "test_event", kUnsetDelegateDebugIntId, 0, 1, nullptr);
  event_tracer_log_profiling_delegate(event_tracer, nullptr, 1, 0, 1, nullptr);
}

TEST(TestEventTracer, SimpleEventTracerTestDelegate) {
  // Call all the EventTracer macro's with a valid pointer to an event tracer
  // and also with a null pointer (to test that the null case works).
  DummyEventTracer dummy;
  std::vector<DummyEventTracer*> dummy_event_tracer_arr = {&dummy, nullptr};
  for (const auto i : c10::irange(dummy_event_tracer_arr.size())) {
    RunSimpleTracerTestDelegate(&dummy);
    RunSimpleTracerTestDelegate(nullptr);
  }
}

TEST(TestEventTracer, SimpleEventTracerTestLogging) {
  using executorch::runtime::internal::event_tracer_log_evalue;
  using executorch::runtime::internal::event_tracer_log_evalue_output;

  EValue test_eval(true);

  {
    // By default there should be no logging enabled.
    DummyEventTracer dummy;
    event_tracer_log_evalue(&dummy, test_eval);
    EXPECT_EQ(dummy.logged_evalue().toBool(), false);
  }

  {
    // Enable only program outputs to be logged. So event_tracer_log_evalue
    // should have no effect but event_tracer_log_evalue_output should work.
    DummyEventTracer dummy;
    dummy.set_event_tracer_debug_level(
        EventTracerDebugLogLevel::kProgramOutputs);
    event_tracer_log_evalue(&dummy, test_eval);
    EXPECT_EQ(dummy.logged_evalue().toBool(), false);
    event_tracer_log_evalue_output(&dummy, test_eval);
    EXPECT_EQ(dummy.logged_evalue().toBool(), true);
    EXPECT_EQ(dummy.logged_evalue_type(), LoggedEValueType::kProgramOutput);
  }

  {
    // Enable all outputs to be logged. So event_tracer_log_evalue and
    // event_tracer_log_evalue_output should both work.
    DummyEventTracer dummy;
    dummy.set_event_tracer_debug_level(
        EventTracerDebugLogLevel::kIntermediateOutputs);
    event_tracer_log_evalue(&dummy, test_eval);
    EXPECT_EQ(dummy.logged_evalue().toBool(), true);
    EXPECT_EQ(
        dummy.logged_evalue_type(), LoggedEValueType::kIntermediateOutput);
    dummy.reset_logged_value();
    event_tracer_log_evalue_output(&dummy, test_eval);
    EXPECT_EQ(dummy.logged_evalue().toBool(), true);
    EXPECT_EQ(dummy.logged_evalue_type(), LoggedEValueType::kProgramOutput);
  }

  // Test with nullptr's to make sure it goes through smoothly.
  event_tracer_log_evalue(nullptr, test_eval);
  event_tracer_log_evalue_output(nullptr, test_eval);
}

// TODO(T163645377): Add more test coverage to log and verify events passed into
// DummyTracer.
TEST(TestEventTracer, EventTracerProfileOpControl) {
  DummyEventTracer dummy;
  // Op profiling is enabled by default. Test that it works.
  {
    {
      executorch::runtime::internal::EventTracerProfileOpScope
          event_tracer_op_scope(&dummy, "ExampleOpScope");
      EXPECT_EQ(strcmp(dummy.get_event_name(), "ExampleOpScope"), 0);
    }
    EXPECT_EQ(strcmp(dummy.get_event_name(), ""), 0);

    // Normal profiling should still work.
    {
      executorch::runtime::internal::EventTracerProfileMethodScope
          event_tracer_profiler_scope(&dummy, "ExampleProfilerScope");
      EXPECT_EQ(strcmp(dummy.get_event_name(), "ExampleProfilerScope"), 0);
    }

    dummy.set_event_tracer_profiling_level(
        executorch::runtime::EventTracerProfilingLevel::kProfileMethodOnly);

    // Op profiling should be disabled now.
    {
      executorch::runtime::internal::EventTracerProfileOpScope
          event_tracer_op_scope(&dummy, "ExampleOpScope");
      EXPECT_EQ(strcmp(dummy.get_event_name(), ""), 0);
    }

    // Normal profiling should still work.
    {
      executorch::runtime::internal::EventTracerProfileMethodScope
          event_tracer_profiler_scope(&dummy, "1ExampleProfilerScope");
      EXPECT_EQ(strcmp(dummy.get_event_name(), "1ExampleProfilerScope"), 0);
    }
  }
}
