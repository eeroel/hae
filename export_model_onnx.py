# from https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=USb4JXZHxqTA
from txtai.pipeline import HFOnnx
import tokenizers
import os
import pickle

onnx = HFOnnx()
MODEL_NAME = "all-MiniLM-L6-v2"

# This model needs tokenizer applied before
# TODO: pre-process/optimize before quantization, see https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
embeddings = onnx(f"{MODEL_NAME}", "pooling", f"{MODEL_NAME}.onnx", quantize=True)
