// TODO:
// - chunking
// - overlap-averaging of too large chunks
// - highlighting logic
// - cli
// - json i/o

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
#include <numeric>
#include <regex>

#include <thread>
#include <future>

#include "hae.h"
#include "argparse.hpp"

// #include "model.h" // TODO uncomment

using tokenizers::Tokenizer;

// calculate cosine similarity between two float vectors:
float cos_sim(const std::vector<float> &a, const std::vector<float> &b)
{
  assert(a.size() == b.size());
  auto dot = 0.f;
  auto norm_a = 0.f;
  auto norm_b = 0.f;

  for (int i = 0; i < a.size(); ++i)
  {
    dot += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::pair<std::vector<int>, std::vector<float>> calc_distances(
    const std::vector<float> &query,
    const std::vector<std::vector<float>> &vectors)
{
  // return ranking using cosine similarities, and the corresponding similarity scores
  auto distances = std::vector<float>(vectors.size(), 0.f);
  for (int i = 0; i < vectors.size(); ++i)
  {
    distances[i] = cos_sim(query, vectors[i]);
  }
  // sort the indices by distance
  auto indices = std::vector<int>(distances.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](const int &a, const int &b)
            { return distances[a] > distances[b]; });
  return std::make_pair(indices, distances);
};


// a function to read all data from stdin into a string:
std::string read_all()
{
    std::stringstream ss;
    std::string s;
    while (std::getline(std::cin, s)) {
        ss << s << '\n';
    }
    return ss.str();
}

// a function to split a string into a vector by a double line break (should handle crlf as well)
std::vector<std::string> split_chunks(const std::string &s)
{
    std::regex re("\r?\n\r?\n");
    return { std::sregex_token_iterator(s.cbegin(), s.cend(), re, -1), {} };
}

std::vector<std::string> combine_chunks(std::vector<std::string> &chunks, int64_t min_size) {
  // greedily combine successive chunks so that the resulting chunks are at least `min_size` long
  std::vector<std::string> combined;
  auto it = chunks.begin();
  while (it != chunks.end()) {
    int64_t size = (*it).size();
    if (size < min_size) {
      // skip this chunk, and the next one as well if it's short too
      for (int i = 0; i < 2 && it != chunks.end(); ++i, ++it) {
        size += (*it).size();
        if (size >= min_size) break;
      }
    } else {
      combined.push_back(*it);
      it = chunks.erase(it);
    }
  }
  return combined;
}

std::vector<std::string> split_sentences(const std::string& text) {
    std::regex wiki_citation_re("(\\^\\[[0-9]+\\])*");
    size_t prev = 0;
    std::vector<std::string> sentences;

    for (auto x = std::sregex_iterator(text.begin(), text.end(), wiki_citation_re);
         x != std::sregex_iterator(); ++x) {
        auto span = x->position() + x->length();
        sentences.push_back(text.substr(prev, span - prev));
        prev = span;
    }

    if (prev < text.size()) {
        sentences.push_back(text.substr(prev));
    }

    return sentences;
}


std::vector<std::vector<int>> build_inputs(const std::string& d, std::unique_ptr<tokenizers::Tokenizer> tok) {
    auto input_size = 128; // TODO take from header
    auto text_prefix = ""; // TODO same
    
    std::vector<int> enc = tok->Encode(text_prefix + d);
    std::vector<int> tokens(enc.begin(), enc.end());

    // max input size handling: average overlapping chunks
    int64_t num_tokens = tokens.size();

    std::vector<std::vector<int>> inputs;

    if (num_tokens % input_size == 0) {
        int64_t start = 0;
        inputs.push_back({tokens.begin() + start, tokens.begin() + start + input_size});
    } else {
        for (int64_t start = 0; start < num_tokens; start += static_cast<int>(input_size*0.9)) {
            if (start >= num_tokens) {
                break;
            }
            inputs.push_back({tokens.begin() + start, tokens.begin() + std::min(start + input_size, num_tokens)});
        }
    }

    return inputs;
}


/*
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
}*/

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape)
{
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser parser("hae");
    // Assuming `parser` is an existing object
    parser.add_argument("query").help("The query");
    parser.add_argument("-n").help("Number of results to return").scan<'i',int>().default_value(5);
    parser.add_argument("--num_candidates").help("Number of candidate chunks() to consider, default is 5*n").scan<'i',int>().default_value(-1);
    parser.add_argument("-ml", "--min_length").help("Minimum chunk length in characters").scan<'i',int>().default_value(300);
    parser.add_argument("-j", "--json").help("Output in JSON format").default_value(false).implicit_value(true);
    parser.add_argument("-hl", "--highlight_only").help("Display only highlights").default_value(false).implicit_value(true);
    try {
      parser.parse_args(argc, argv);   // Example: ./main --input_files config.yml System.xml
    }
    catch (const std::exception& err) {
      std::cerr << err.what() << std::endl;
      std::cerr << parser;
      std::exit(1);
    }

    int num_results = parser.get<int>("-n");
    int top_k = parser.get<int>("--num_candidates") ? parser.get<int>("--num_candidates") : 5 * num_results;
    std::string query = parser.get("query");
    int min_length = parser.get<int>("--min_length");
    bool output_json = parser.get<bool>("--json");
    bool highlight_only = parser.get<bool>("--highlight_only");



  // Read blob from file.
  // auto tokenizer_json = LoadBytesFromFile("tokenizer.json"); // TODO embed this!

  // Note: all the current factory APIs takes in-memory blob as input.
  // This gives some flexibility on how these blobs can be read.
  auto tok = Tokenizer::FromBlobJSON(tokenizer_json);

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  const auto &api = Ort::GetApi();

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(3);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
  const wchar_t *model_path = L"all-MiniLM-L6-v2.onnx";
#else
  const char *model_path = "all-MiniLM-L6-v2.onnx";
#endif

  // NOTE: for dev we load model from file, but for release we should embed it
  Ort::Session session(env, model_path, session_options);
  // Ort::Session session(env, all_MiniLM_L6_v2_onnx, sizeof(all_MiniLM_L6_v2_onnx), session_options); // TODO: uncomment
  //  print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;

  for (std::size_t i = 0; i < session.GetInputCount(); i++)
  {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
  }

  // print name/shape of outputs
  std::vector<std::string> output_names;
  for (std::size_t i = 0; i < session.GetOutputCount(); i++)
  {
    output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
  }

  // pass data through model
  std::vector<const char *> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                 [&](const std::string &str)
                 { return str.c_str(); });

  std::vector<const char *> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                 [&](const std::string &str)
                 { return str.c_str(); });

  auto input_shape = 128;  // TODO store somewhere else?
  auto output_shape = 384; // TODO store somewhere else?

  auto ids = tok->Encode("whales are dangerous");

  std::vector<int64_t> token_ids;
  std::transform(ids.begin(), ids.end(),
                 std::back_inserter(token_ids),
                 [](int32_t i)
                 { return static_cast<int64_t>(i); });
  std::vector<int64_t> zeros(token_ids.size(), 0);
  std::vector<int64_t> ones(token_ids.size(), 1);

  std::vector<std::thread> threads;
  std::vector<std::future<std::vector<Ort::Value>>> futures;

  for (int i = 0; i < 10; ++i)
  {
    futures.emplace_back(
        std::async([&]()
                   {
        Ort::RunOptions runOptions;

        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(vec_to_tensor(token_ids, {1, input_shape})); // token ids
        input_tensors.emplace_back(vec_to_tensor(ones, {1, input_shape})); // attention mask
        input_tensors.emplace_back(vec_to_tensor(zeros, {1, input_shape})); // token type ids

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                          input_names_char.size(), output_names_char.data(), output_names_char.size());
        return output_tensors; }));
  }

  std::vector<std::vector<float>> vectors;
  for (auto &fut : futures)
  {
    auto tensor = fut.get();
    std::vector<float> v;
    float *floatarr = tensor.front().GetTensorMutableData<float>();
    // score the model, and print scores for first 5 classes
    for (int i = 0; i < output_shape; i++)
    {
      v.emplace_back(floatarr[i]);
    }
    vectors.emplace_back(v);
  }

  auto query_ids = tok->Encode(query);

  std::vector<int64_t> query_token_ids;
  std::transform(query_ids.begin(), query_ids.end(),
                 std::back_inserter(query_token_ids),
                 [](int32_t i)
                 { return static_cast<int64_t>(i); });
  std::vector<int64_t> query_zeros(query_token_ids.size(), 0);
  std::vector<int64_t> query_ones(query_token_ids.size(), 1);

  Ort::RunOptions runOptions;

  std::vector<Ort::Value> input_tensors;
  input_tensors.emplace_back(vec_to_tensor(query_token_ids, {1, input_shape})); // token ids
  input_tensors.emplace_back(vec_to_tensor(query_ones, {1, input_shape}));      // attention mask
  input_tensors.emplace_back(vec_to_tensor(query_zeros, {1, input_shape}));     // token type ids

  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                    input_names_char.size(), output_names_char.data(), output_names_char.size());
  std::vector<float> v;
  float *floatarr = output_tensors.front().GetTensorMutableData<float>();
  // score the model, and print scores for first 5 classes
  for (int i = 0; i < output_shape; i++)
  {
    v.emplace_back(floatarr[i]);
  }

  auto foo = calc_distances(v, vectors);
  for (int i = 0; i < foo.first.size(); i++)
  {
    std::cout << foo.first[i] << ", " << foo.second[i] << std::endl;
  }
}
