#define HAE_VERSION "0.1.3"

#include <tokenizers_cpp.h>
#include <onnxruntime_cxx_api.h>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

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

#include "build/tokenizer.h"
#include "argparse.hpp"

#include "build/model.h"

using tokenizers::Tokenizer;

struct InputDocument {
  std::string title;
  std::string content;
};

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
  std::sort(distances.begin(), distances.end(), [&](const float &a, const float &b)
            { return a > b; });
  return std::make_pair(indices, distances);
};


// a function to read all data from stdin into a string:
std::string read_all()
{
    std::ios::sync_with_stdio(false);
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

    std::string buffer = "";
    for (size_t i = 0; i < chunks.size(); ++i) {
        buffer += chunks[i] + "\n";
        // If the chunk has multiple lines, just append it
        if (buffer.length() > min_size) {
            combined.push_back(buffer.substr(0, buffer.length() - 1));
            buffer.clear();
            continue;
        }
    }
    if (!buffer.empty()) {
        combined.push_back(buffer.substr(0, buffer.length() - 1));
    }

    return combined;
}

std::vector<std::string> split_sentences(const std::string& text) {
    std::string wiki_citation_re = "(\\^\\[[0-9]+\\])*";
    std::regex full_re(":\\n" + wiki_citation_re + "|[.!?]" + wiki_citation_re + "\\s");
    size_t prev = 0;
    std::vector<std::string> sentences;

    for (auto x = std::sregex_iterator(text.begin(), text.end(), full_re);
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


std::vector<int> tokenize(const std::string &s, std::unique_ptr<tokenizers::Tokenizer> &tok) {
   // TODO: see if this can be improved:
   // we prepend the special [CLS] token here because of a hunch
   // that it's required for the model (first token gets ignored)
   // Might be that with a different ONNX export this is not required
   auto tokens = tok->Encode("[CLS]" + s);
   int i = tokens.size();
   while (i > 0 && tokens[i-1] == 0) {
      --i;
   }
   return std::vector<int>(tokens.begin(), tokens.begin() + i);
}

std::vector<std::vector<int>> build_inputs(const std::string &d, std::unique_ptr<tokenizers::Tokenizer> &tok) {
    auto text_prefix = ""; // TODO same
    
    std::vector<int> enc = tokenize(text_prefix + d, tok);
    std::vector<int> tokens(enc.begin(), enc.end());

    // max input size handling: average overlapping chunks
    int64_t num_tokens = tokens.size();

    std::vector<std::vector<int>> inputs;

    if (num_tokens < input_size) {
      inputs.push_back({tokens.begin(), tokens.begin() + num_tokens});
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

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape)
{
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

std::vector<float> tensor_to_vector(std::vector<Ort::Value> &tensor) {
  std::vector<float> v;
  float *floatarr = tensor.front().GetTensorMutableData<float>();

  for (int i = 0; i < output_size; i++)
  {
    v.emplace_back(floatarr[i]);
  }
  return v;
}

std::vector<Ort::Value> run_onnx(
  std::vector<int> &input,
  std::vector<const char *> &input_names_char,
  std::vector<const char *> &output_names_char,
  Ort::Session &session
) {
    Ort::RunOptions runOptions;
    long long input_shape = input.size();

    std::vector<int64_t> token_ids;
    std::transform(input.begin(), input.end(),
                  std::back_inserter(token_ids),
                  [](int32_t i)
                  { return static_cast<int64_t>(i); });
    std::vector<int64_t> zeros(input_shape, 0);
    std::vector<int64_t> ones(input_shape, 1);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(vec_to_tensor(token_ids, {1, input_shape})); // token ids
    input_tensors.emplace_back(vec_to_tensor(ones, {1, input_shape})); // attention mask
    input_tensors.emplace_back(vec_to_tensor(zeros, {1, input_shape})); // token type ids

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                      input_names_char.size(), output_names_char.data(), output_names_char.size());
    return output_tensors;
}

std::vector<std::vector<std::vector<int>>> tokenize_all(
  std::vector<std::string> &input,
  std::unique_ptr<tokenizers::Tokenizer> &tok
) {
  std::vector<std::vector<std::vector<int>>> inputs;
  for (auto &c : input) {
      auto input = build_inputs(c, tok);
      inputs.push_back(input);
  }
  return inputs;
}

std::vector<std::vector<float>> vectorize_all_batch(
  std::vector<std::vector<std::vector<int>>> &inputs,
  std::vector<const char *> &input_names_char,
  std::vector<const char *> &output_names_char,
  Ort::Session &session
  ) {
  std::vector<std::vector<std::future<std::vector<Ort::Value>>>> futures;

  for (auto &sub_chunks: inputs)
  {
    std::vector<std::future<std::vector<Ort::Value>>> subfutures;
    for (auto &ids: sub_chunks)
    {
      subfutures.emplace_back(
          std::async(std::launch::async, [&]()
            {
              return run_onnx(ids, input_names_char, output_names_char, session);
            })
        );
    }
    futures.emplace_back(std::move(subfutures));
  }

  std::vector<std::vector<float>> vectors;
  for (auto &subfut : futures)
  {
    std::vector<float> averaged(output_size, 0);
    for (auto &fut : subfut)
    {
      auto tensor = fut.get();
      auto v = tensor_to_vector(tensor);

      // add v elementwise to averaged
      std::transform(v.begin(), v.end(), averaged.begin(), averaged.begin(), std::plus<float>());
    }
    // divide averaged by number of elements in subfutures
    for (int i = 0; i < output_size; i++)
    {
      averaged[i] /= subfut.size();
    }
    vectors.emplace_back(averaged);
  }
  return vectors;
}

std::vector<std::vector<float>> vectorize_all(
  std::vector<std::vector<std::vector<int>>> &inputs,
  std::vector<const char *> &input_names_char,
  std::vector<const char *> &output_names_char,
  Ort::Session &session
  ) {
    std::vector<std::vector<float>> results;
    // iterate over `inputs` in batches of batch_size:
    // the idea here is to limit the amount of threads created by async.
    // not optimal but seems to work OK
    int batch_size = 128; // upper limit on number of threads used
    auto num_batches = static_cast<int>(inputs.size()/batch_size) + 1;
    for (int i = 0; i < num_batches; ++i) {
      // take next batch:
      std::vector<std::vector<std::vector<int>>> batch;
      if (i < num_batches-1) {
        std::vector<std::vector<std::vector<int>>> batch(inputs.begin() + i*batch_size, inputs.begin() + (i+1)*batch_size);
        auto batch_result = vectorize_all_batch(batch, input_names_char, output_names_char, session);
        for (auto &res : batch_result)
        {
            results.push_back(res);
        }
      } else {
        std::vector<std::vector<std::vector<int>>> batch(inputs.begin() + i*batch_size, inputs.end());
        auto batch_result = vectorize_all_batch(batch, input_names_char, output_names_char, session);
        for (auto &res : batch_result)
        {
            results.push_back(res);
        }
      }
    }
    return results;
  }

int main(int argc, char *argv[])
{
    argparse::ArgumentParser parser("hae", HAE_VERSION);
    // Assuming `parser` is an existing object
    parser.add_argument("query").help("The query");
    parser.add_argument("-n").help("Number of results to return").scan<'i',int>().default_value(5);
    parser.add_argument("--num_candidates").help("Number of candidate chunks to consider, default is 5*n or at least 30").scan<'i',int>().default_value(-1);
    parser.add_argument("-ml", "--min_length").help("Minimum chunk length in characters").scan<'i',int>().default_value(512);
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
    int top_k = parser.get<int>("--num_candidates") > 0 ? parser.get<int>("--num_candidates") : std::max(30, 5 * num_results);
    std::string query = parser.get("query");
    int min_length = parser.get<int>("--min_length");
    bool output_json = parser.get<bool>("--json");
    bool highlight_only = parser.get<bool>("--highlight_only");

    // Note: all the current factory APIs takes in-memory blob as input.
    // This gives some flexibility on how these blobs can be read.
    auto tok = Tokenizer::FromBlobJSON(std::string((const char*)tokenizer_json, tokenizer_json_len));

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  const auto &api = Ort::GetApi();

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session session(env, all_MiniLM_L6_v2_onnx, sizeof(all_MiniLM_L6_v2_onnx), session_options);
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

  std::string input = read_all(); // from stdin

  // try parsing as newline delimited json. if it fails, do chunking
  std::vector<std::string> combined;
  bool json_input = true;

  std::vector<std::shared_ptr<rapidjson::Document>> json_docs;
  std::vector<InputDocument> docs;
  
  try {
    std::vector<InputDocument> docs_candidate;
    std::stringstream ss(input);
    std::string line;
    while (std::getline(ss, line)) {
        if (line == "") {
          break;
        }
        auto d = std::shared_ptr<rapidjson::Document>(new rapidjson::Document());
        d->Parse(line.c_str());
        if (d->IsNull()) {
          throw 1;
        }
        json_docs.emplace_back(d);
    }
    
    for (int i = 0; i < json_docs.size(); ++i) {
      rapidjson::Document& d = *json_docs[i];

      // content is required
      if (!d.HasMember("content") || !d["content"].IsString()) {
        throw 1;
      }
      std::string title = d.HasMember("title") && d["title"].IsString() ? std::string(d["title"].GetString()) : "";
      std::string content = std::string(d["content"].GetString());

      combined.push_back( title + "\\n" + content );
      docs_candidate.push_back(InputDocument{title, content});
    }
    docs = docs_candidate;
  } catch (int e) {
    json_input = false;
    // process non-json input
    auto chunked = split_chunks(input);
    combined = combine_chunks(chunked, min_length);
    for (int i = 0; i < combined.size(); ++i) {
      docs.push_back(InputDocument{"", combined[i]});
    }
  }

  auto tokens = tokenize_all(combined, tok);
  auto vectors = vectorize_all(tokens, input_names_char, output_names_char, session);
  auto query_ids = tokenize(query, tok);
  // truncate to max number of tokens
  if (query_ids.size() > input_size)
  {
    query_ids.resize(input_size);
  }

  auto output_tensors = run_onnx(query_ids, input_names_char, output_names_char, session);
  auto v_query = tensor_to_vector(output_tensors);
  auto distances = calc_distances(v_query, vectors);

  // Calculate highlights
  std::vector<float> highlight_scores;
  std::vector<std::string> highlights;
  std::vector<size_t> selected_indices;

  size_t total_chunks = std::min(static_cast<size_t>(top_k), distances.first.size());

  for (size_t i = 0; i < total_chunks; ++i) {
      auto chunk = combined[distances.first[i]];
      selected_indices.push_back(distances.first[i]);
  }

  std::vector<std::string> sentences{};
  std::vector<int> chunk_ids;

  for (size_t i = 0; i < total_chunks; ++i) {
      auto chunk = combined[selected_indices[i]];
      // Do another round of search within the text, splitting on sentences
      for (const auto& x : split_sentences(chunk)) {
        std::string s = x;
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
        }));
        
        if (s.size() > 3) { // TODO better logic/do size filtering somewhere else?
          sentences.push_back(x);
          chunk_ids.push_back(i);
        }
      }
  }

  auto sentence_tokens = tokenize_all(sentences, tok);
  std::vector<std::vector<float>> sentence_vectors = vectorize_all(sentence_tokens, input_names_char, output_names_char, session);

  // iterate sentence_vectors grouped by chunk_ids:
  for (size_t i = 0; i < total_chunks; ++i) {
    std::vector<std::vector<float>> vecs;
    std::vector<std::string> sents;
    for (int j = 0; j < sentence_vectors.size(); ++j) {
        if (chunk_ids[j] == i) {
            vecs.push_back(sentence_vectors[j]);
            sents.push_back(sentences[j]);
        }
    }

    if (vecs.size() == 0) {
      // if there's not a single sentence to highlight just push a small value so it will
      // be at the tail
      highlight_scores.push_back(-10000);
      highlights.push_back("");
    } else {
      auto dists = calc_distances(v_query, vecs);
      highlight_scores.push_back(dists.second[0]);
      highlights.push_back(sents[dists.first[0]]);
    }
  }

  // sort highlight_scores and reorder highlights by the same sort:
  std::vector<std::pair<float, int>> sorted_highlight_scores;
  for (int i = 0; i < highlight_scores.size(); ++i) {
    sorted_highlight_scores.push_back(std::make_pair(highlight_scores[i], i));
  }

  std::sort(sorted_highlight_scores.begin(), sorted_highlight_scores.end(), [](auto &left, auto &right) {
    return left.first > right.first;
  });

  std::vector<std::string> sorted_highlights;
  std::vector<size_t> sorted_selected_indices;

  for (int i = 0; i < highlight_scores.size(); ++i) {
    sorted_highlights.push_back(highlights[sorted_highlight_scores[i].second]);
    sorted_selected_indices.push_back(selected_indices[sorted_highlight_scores[i].second]);
  }

  size_t result_count = std::min(docs.size(), static_cast<size_t>(num_results));
  for (int i = 0; i < result_count; i++) {
    auto title = docs[sorted_selected_indices[i]].title;
    auto content = docs[sorted_selected_indices[i]].content;
    auto highlight = sorted_highlights[i];

    if (output_json || json_input) {
      std::shared_ptr<rapidjson::Document> d;
      if (json_docs.size() == 0) {
        d = std::shared_ptr<rapidjson::Document>(new rapidjson::Document());
        d->SetObject();
      } else {
        d = json_docs[sorted_selected_indices[i]];
      }
      rapidjson::Value& v = *d;

      rapidjson::Value v_title; // TODO only write highlight if other fields exist
      if (json_docs.size() == 0) {
        v_title.SetString(rapidjson::StringRef(title.c_str()));
        v.AddMember("title", v_title, d->GetAllocator());
        rapidjson::Value v_content;
        v_content.SetString(rapidjson::StringRef(content.c_str()));
        v.AddMember("content", v_content, d->GetAllocator());
      }
      rapidjson::Value v_highlight;
      v_highlight.SetString(rapidjson::StringRef(highlight.c_str()));
      v.AddMember("highlight", v_highlight, d->GetAllocator());

      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      d->Accept(writer);
      std::cout << buffer.GetString() << std::endl;
    } else {
      if (highlight_only) {
          std::cout << highlight << std::endl << std::endl;
      } else {
        size_t pos = 0;
        pos = content.find(highlight, pos);
        auto highlighted = content;
        if (pos != std::string::npos) {
            highlighted = content.replace(pos, highlight.length(), std::string("\033[1m" + highlight + "\033[0m"));
        }
        std::cout << highlighted << std::endl << std::endl;
      }
    }
  }
}
