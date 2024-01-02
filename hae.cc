/*
    Modified from https://github.com/microsoft/onnxruntime-inference-examples
    Original license:

    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
*/

#include <tokenizers_cpp.h>
#include <onnxruntime_cxx_api.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using tokenizers::Tokenizer;

std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}


template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

void infer(std::vector<int>& ids) {
  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  //Ort::AllocatorWithDefaultOptions allocator;
  /* TODO call session.Run with
  input_names – Array of null terminated strings of length input_count that is the list of input names
  input_values – Array of Value objects of length input_count that is the list of input values
  input_count – Number of inputs (the size of the input_names & input_values arrays)
  output_names – Array of C style strings of length output_count that is the list of output names
  output_count – Number of outputs (the size of the output_names array)
  */

  //session.Run(
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int main(int argc, char* argv[]) {
  auto start = std::chrono::high_resolution_clock::now();

  // Read blob from file.
  auto blob = LoadBytesFromFile("tokenizer.json"); // TODO embed this!

  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobJSON(blob);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  const auto& api = Ort::GetApi();

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(3);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  #ifdef _WIN32
  const wchar_t* model_path = L"all-MiniLM-L6-v2.onnx";
#else
  const char* model_path = "all-MiniLM-L6-v2.onnx";
#endif
  Ort::Session session(env, model_path, session_options);

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;
  
  std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetInputCount(); i++) {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    std::cout << input_names[i] << std::endl;
  }

    // print name/shape of outputs
  std::vector<std::string> output_names;
  std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
    output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
  }

  // pass data through model
  std::vector<const char*> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                 [&](const std::string& str) { return str.c_str(); });

  std::vector<const char*> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                 [&](const std::string& str) { return str.c_str(); });


  auto ids = tok->Encode("This is a test");
  std::vector<int64_t> token_ids;
  std::transform(ids.begin(), ids.end(),
                  std::back_inserter(token_ids),
                  [](int32_t i) { return static_cast<int64_t>(i); });
  std::vector<int64_t> zeros(token_ids.size(), 0);
  std::vector<int64_t> ones(token_ids.size(), 1);

  std::vector<Ort::Value> input_tensors;

  auto input_shape = 128; // TODO get from tokenizer

  input_tensors.emplace_back(vec_to_tensor(token_ids, {1, input_shape})); // token ids
  input_tensors.emplace_back(vec_to_tensor(ones, {1, input_shape})); // attention mask
  input_tensors.emplace_back(vec_to_tensor(zeros, {1, input_shape})); // token type ids

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 384; i++) {
    std::cout << floatarr[i] << ", ";
  }
}
