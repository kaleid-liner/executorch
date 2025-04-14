# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# model sharding with custom op
# preprocess qnn runner src files for llama
if(NOT EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../..)
endif()

include(${EXECUTORCH_ROOT}/build/Utils.cmake)
include(${EXECUTORCH_ROOT}/build/Codegen.cmake)

#
# The `_<target>_srcs` lists are defined by including ${EXECUTORCH_SRCS_FILE}.
#
set(EXECUTORCH_SRCS_FILE
    "${CMAKE_CURRENT_BINARY_DIR}/../../../../../executorch_srcs.cmake"
)

extract_sources(${EXECUTORCH_SRCS_FILE})

include(${EXECUTORCH_SRCS_FILE})

set(_llama_runner__srcs ${_llama_runner__srcs})

list(TRANSFORM _llama_runner__srcs PREPEND "${EXECUTORCH_ROOT}/")
list(FILTER _llama_runner__srcs EXCLUDE REGEX ".*(/runner/).*")

target_include_directories(
  extension_module INTERFACE ${_common_include_directories}
)

list(
  PREPEND
  _llama_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner.h
  ${CMAKE_CURRENT_LIST_DIR}/io_manager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/io_manager.h
)

list(
  APPEND
  _llama_runner__srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../extension/llm/tokenizer/tiktoken.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../models/llama/tokenizer/llama_tiktoken.cpp
)

# build qnn llama runner
if(CMAKE_TOOLCHAIN_IOS
   OR ANDROID
   OR APPLE
)
  # Building a share library on iOS requires code signing On Android we see
  # duplicated registration when using shared lib
  add_library(llama_runner STATIC ${_llama_runner__srcs})
else()
  add_library(llama_runner SHARED ${_llama_runner__srcs})
endif()

target_include_directories(
  llama_runner PUBLIC ${_common_include_directories}
)

# find RE2 for tokenizer, build tiktoken
set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(_pic_flag ${CMAKE_POSITION_INDEPENDENT_CODE})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(
  ${EXECUTORCH_ROOT}/extension/llm/third-party/abseil-cpp
  ${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp
)
add_subdirectory(
  ${EXECUTORCH_ROOT}/extension/llm/third-party/re2
  ${CMAKE_CURRENT_BINARY_DIR}/re2
)
set(CMAKE_POSITION_INDEPENDENT_CODE ${_pic_flag})

target_link_libraries(
  llama_runner
  PUBLIC
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_module
  extension_tensor
  re2::re2
  custom_ops
)
target_compile_options(
  llama_runner PUBLIC ${_common_compile_options}
)
# set_target_properties(
#   llama_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
# )
