#/bin/bash

# TODO: complete pipeline
# - download onnx runtime distribution
# - download tokenizer and model
# - convert model to onnx
# - update tokenizer file as needed (truncation params - can this be done via python?)
# - convert tokenizer and model into hexdumps
# - ???
# - profit!

# build
mkdir -p build
cd build
cmake .. -DONNXRUNTIME_ROOTDIR=${ONNXRUNTIME_ROOTDIR} -DCMAKE_BUILD_TYPE=Release
make -j8
cd ..
