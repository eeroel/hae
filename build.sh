#/bin/bash

# TODO:
# - intel mac build
set -e

# target should be one of:
# osx-arm64
# linux-x64
# osx-x86_64
target=$1

# build
mkdir -p build
cd build

# fetch onnx runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-$target-1.16.3.tgz -O onnxruntime-$target-1.16.3.tgz
tar -xvf ./onnxruntime-$target-1.16.3.tgz

# fetch model
mkdir -p all-MiniLM-L6-v2
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin?download=true -O all-MiniLM-L6-v2/pytorch_model.bin
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json?download=true -O all-MiniLM-L6-v2/tokenizer.json
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json?download=true -O all-MiniLM-L6-v2/config.json

if [[ -z "${RUNNING_IN_DOCKER-}" ]]; then
    python3.10 -m venv create venv && source venv/bin/activate
    pip install -r ../requirements-dev.txt
fi

python ../export_model_onnx.py

xxd -i all-MiniLM-L6-v2.onnx > model.h

input_size=128  # NOTE: model specific config - read from model files if possible
output_size=384
# write tokenizer to header file
echo -e "#pragma once\nauto input_size=$input_size;\nauto output_size=$output_size;\n" > tokenizer.h
# remove truncation and padding from tokenizer config
jq -c 'del(.truncation, .padding)' all-MiniLM-L6-v2/tokenizer.json > tokenizer.json
xxd -i tokenizer.json >> tokenizer.h

if [[ -v RUNNING_IN_DOCKER ]]; then
    # TODO does this do anything? an attempt to fix the flakiness of the build
    source $HOME/.cargo/env && rustup default 1.71.0 && rustup toolchain remove stable
fi

cmake .. -DONNXRUNTIME_ROOTDIR=$PWD/onnxruntime-$target-1.16.3 -DCMAKE_BUILD_TYPE=Release
make -j8

if [[ -z "${RUNNING_IN_DOCKER-}" ]]; then
    deactivate
fi

cd ..

mkdir -p dist
cp build/hae dist
cp LICENSE dist
cp NOTICES dist

if [[ "$target" != "linux-x64" ]]; then
    cp build/onnxruntime-$target-1.16.3/lib/libonnxruntime.1.16.3.dylib dist/
fi

if [[ "$target" == "linux-x64" ]]; then
    cp build/onnxruntime-$target-1.16.3/lib/libonnxruntime.so.1.16.3 dist/
fi