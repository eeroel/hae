# hae

A command line tool for semantic search over text input.

## Features:
- Highlighting matched sentences
- Optional JSON input and output

## Acknowledgements
- This project is only possible due to SentenceTransformers https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- ONNX Runtime for efficient CPU inferencing https://onnxruntime.ai/
- HuggingFace Tokenizers via a C++ wrapper https://github.com/mlc-ai/tokenizers-cpp
- https://github.com/p-ranav/argparse for creating the CLI
- https://rapidjson.org/ for JSON I/O
