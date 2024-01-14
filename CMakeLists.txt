# Example cmake project
cmake_minimum_required(VERSION 3.19)

project(hae C CXX)

set(CMAKE_MACOSX_RPATH TRUE)

# include tokenizer cpp as a sub directory
set(TOKENIZER_CPP_PATH tokenizers-cpp)
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

add_executable(hae hae.cc)

set(ONNXRUNTIME_PATH "${CMAKE_SOURCE_DIR}/build/onnxruntime")

if(APPLE)
      set(DEPS "${ONNXRUNTIME_PATH}/build/MacOS/Release/_deps")
      target_link_directories(hae PRIVATE "${ONNXRUNTIME_PATH}/build/MacOS/Release")
else()
      set(DEPS "${ONNXRUNTIME_PATH}/build/Linux/Release/_deps")
      target_link_directories(hae PRIVATE "${ONNXRUNTIME_PATH}/build/Linux/Release")
endif()

add_custom_command(
    OUTPUT ${DEPS}/libonnxruntime_session.a
    WORKING_DIRECTORY ${ONNXRUNTIME_PATH}
    COMMAND ./build.sh --config Release --parallel --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER --allow_running_as_root
)

add_custom_target(
    onnxruntime
    DEPENDS ${DEPS}/libonnxruntime_session.a
)

add_dependencies(hae onnxruntime)

# TODO: remove
if(APPLE)
    set_target_properties(hae PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,@loader_path/,-framework,Foundation")
else()
    set_target_properties(hae PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        LINK_FLAGS "-Wl,-rpath,$ORIGIN/")
endif()

include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

#onnxruntime providers
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)

if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
endif()
if(onnxruntime_USE_DML)
  message("Enabling DML")
  add_definitions(-DUSE_DML)
endif()

target_include_directories(hae PRIVATE ${TOKENIZER_CPP_PATH}/include)
target_include_directories(
  hae
  PRIVATE
  ${ONNXRUNTIME_PATH}/include/onnxruntime/core/session
  ${ONNXRUNTIME_PATH}/include/onnxruntime/core/providers/cpu
  ${ONNXRUNTIME_PATH}/include/onnxruntime/core/framework
  )
# You can link tokenizers_cpp, it will automatically link tokenizers_c
# and sentencepiece libary
target_link_libraries(hae PRIVATE tokenizers_cpp)


# link all the static libraries produced by onnxruntime build
# see https://github.com/microsoft/onnxruntime/issues/15032
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/base/)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/container/)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/hash)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/numeric)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/profiling/)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/strings/)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/synchronization/)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/time/)
target_link_directories(hae PUBLIC ${DEPS}/abseil_cpp-build/absl/types/)
target_link_directories(hae PUBLIC ${DEPS}/flatbuffers-build/)
target_link_directories(hae PUBLIC ${DEPS}/google_nsync-build/)
target_link_directories(hae PUBLIC ${DEPS}/onnx-build/)
target_link_directories(hae PUBLIC ${DEPS}/protobuf-build/)
target_link_directories(hae PUBLIC ${DEPS}/pytorch_clog-build/)
target_link_directories(hae PUBLIC ${DEPS}/pytorch_cpuinfo-build/)
target_link_directories(hae PUBLIC ${DEPS}/re2-build/)

target_link_libraries(
  hae
  PRIVATE
  libonnxruntime_session.a
  libonnxruntime_optimizer.a
  libonnxruntime_providers.a
  libonnxruntime_util.a
  libonnxruntime_framework.a
  libonnxruntime_graph.a
  libonnxruntime_mlas.a
  libonnxruntime_common.a
  libonnxruntime_flatbuffers.a
  libonnx_test_data_proto.a
  libabsl_base.a
  libabsl_log_severity.a
  libabsl_malloc_internal.a
  libabsl_raw_logging_internal.a
  libabsl_spinlock_wait.a
  libabsl_throw_delegate.a
  libabsl_hashtablez_sampler.a
  libabsl_raw_hash_set.a
  libabsl_hash.a
  libabsl_city.a
  libabsl_low_level_hash.a
  libabsl_int128.a
  libabsl_exponential_biased.a
  libabsl_cord.a
  libabsl_cord_internal.a
  libabsl_cordz_functions.a
  libabsl_cordz_handle.a
  libabsl_cordz_info.a
  libabsl_strings.a
  libabsl_strings_internal.a
  libabsl_graphcycles_internal.a
  libabsl_synchronization.a
  libabsl_civil_time.a
  libabsl_time.a
  libabsl_time_zone.a
  libabsl_bad_optional_access.a
  libabsl_bad_variant_access.a
  libflatbuffers.a
  libnsync_cpp.a
  libonnx.a
  libonnx_proto.a
  libprotobuf-lite.a
  #libprotobuf.a
  #libprotoc.a
  libcpuinfo.a
  libclog.a
  libre2.a
  pthread
  #atomic
  dl
)