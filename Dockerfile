FROM python:3.10-slim
RUN apt update && apt install -y wget jq && apt install -y g++
RUN apt install -y xxd

RUN apt install -y curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain 1.71.0 -y

RUN apt install -y clang
RUN apt install -y git
RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-linux-x86_64.sh && sh ./cmake-3.28.1-linux-x86_64.sh --skip-license --prefix=/usr/
RUN apt install make

ADD requirements-dev.txt /app/requirements-dev.txt
RUN pip install -r /app/requirements-dev.txt

ADD build.sh /app/build.sh
ADD CMakeLists.txt /app/CMakeLists.txt
ADD export_model_onnx.py /app/export_model_onnx.py
ADD hae.cc /app/hae.cc
ADD argparse.hpp /app/argparse.hpp
ADD rapidjson /app/rapidjson
ADD tokenizers-cpp /app/tokenizers-cpp
ADD onnxruntime /app/onnxruntime
ADD CMakeLists.txt /app/CMakeLists.txt
ADD LICENSE /app/LICENSE
ADD NOTICES /app/NOTICES
ENV RUNNING_IN_DOCKER=1

WORKDIR /app
