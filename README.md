# hae

A command line tool for semantic search over text input.

Example:
```
~> man ls | hae "how to show file sizes" -n 1 -hl
   The Long Format
     If the -l option is given, the following information is displayed for
     each file: file mode, number of links, owner name, group name, number of
     bytes in the file, abbreviated month, day-of-month file was last
     modified, hour file last modified, minute file last modified, and the
     pathname.
```

Try it with some text on the clipboard, an RSS feed converted to JSON, or with a cup of coffee ☕ and a nice book 📖 from Project Gutenberg!

_Disclaimer: This is alpha quality software so use it at your own risk!_

## Features:
- Highlighting best-matching sentences, which makes it easier to quickly evaluate the results.
- Optional JSON input and output. Input is automatically interpreted as JSON if it is in the "JSON lines" format, one object per line, and each object contains a field `content`. If a `title` field is present, it will also be used for the search. **Only JSON lines format is supported for input.**

## Installation
For Linux and Apple Silicon Macs you can download a prebuilt binary. Otherwise you can build from source.

## Build from source
The repo contains a build script that also downloads the embedding model and converts it to ONNX format using Python libraries. So you will need some tools installed. Alternatively, for Linux, you can use the Docker image to build (see below).

- cmake
- clang
- rust (required for Tokenizers dependency)
- wget (for fetching ONNX runtime and model files)
- xxd (for embedding model files in headers)
- jq (for preprocessing the tokenizer config file)
- Python (3.10)

If you are building for Linux, please run the build script with a Python virtual environment activated.

`./build.sh $ARCH` where `$ARCH` is one of the following: osx-arm64, linux-x64, osx-x86_64

The application and the ONNX runtime dynamic library required to run it will be found under `./dist`.

### Docker
```
mkdir dist-linux-x64
docker build --platform linux/amd64 -t hae:latest .
docker run -it -v./dist-linux-x64:/app/dist --platform linux/amd64 hae bash -c './build.sh linux-x64'
```

## Acknowledgements
- This project is only possible due to SentenceTransformers https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- ONNX Runtime for efficient CPU inferencing https://onnxruntime.ai/
- HuggingFace Tokenizers via a C++ wrapper https://github.com/mlc-ai/tokenizers-cpp
- https://github.com/p-ranav/argparse for creating the CLI
- https://rapidjson.org/ for JSON I/O
