
# Example cmake project
cmake_minimum_required(VERSION 3.19)
project(tokenizers_cpp_example C CXX)

# include tokenizer cpp as a sub directory
set(TOKENIZER_CPP_PATH tokenizers-cpp)
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

add_executable(hae hae.cc)

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
option(ONNXRUNTIME_ROOTDIR "onnxruntime root dir")
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
target_include_directories(hae PUBLIC "${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

#target_link_directories(hae PUBLIC "${ONNXRUNTIME_ROOTDIR}/lib")

# TODO don't hardcode lib
target_link_libraries(hae PUBLIC "${ONNXRUNTIME_ROOTDIR}/lib/libonnxruntime.dylib")

if(WIN32)
  add_library(wil INTERFACE)

  FetchContent_Declare(
      microsoft_wil
      URL https://github.com/microsoft/wil/archive/refs/tags/v1.0.220914.1.zip
    )
  FetchContent_Populate(microsoft_wil)
  target_include_directories(wil INTERFACE ${microsoft_wil_SOURCE_DIR}/include)
  set(WIL_LIB wil)
endif()


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

# Windows might have an onnxruntime.dll in the system directory so it's more robust to manually copy the dlls to
# the output dir. Define a function to do so. This is called from the cmake file in the subdirectories.
function(copy_ort_dlls target_name)
  if (MSVC)
      file(GLOB ORT_DLLS ${ONNXRUNTIME_ROOTDIR}/bin/*.dll)
      foreach(ORT_DLL ${ORT_DLLS})
          add_custom_command(TARGET ${target_name} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL}  $<TARGET_FILE_DIR:${target_name}>)
      endforeach()
  endif()
endfunction()

target_include_directories(hae PRIVATE ${TOKENIZER_CPP_PATH}/include)

# You can link tokenizers_cpp, it will automatically link tokenizers_c
# and sentencepiece libary
target_link_libraries(hae PRIVATE tokenizers_cpp)
