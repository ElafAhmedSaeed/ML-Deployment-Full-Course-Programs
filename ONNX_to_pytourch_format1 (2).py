
# ================================
# Step 1: TensorFlow → ONNX
# ================================

import tensorflow as tf
import tf2onnx
import torch
import onnx
from onnx2pytorch import ConvertModel

# Build a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(3, activation="softmax")
])

# Convert TensorFlow model to ONNX
spec = (tf.TensorSpec((None, 4), tf.float32, name="input"),)

onnx_model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path="model.onnx"
)

print("ONNX model saved as model.onnx")

# ================================
# Step 2: ONNX → PyTorch
# ================================

# 🔹 Load ONNX model properly
onnx_model = onnx.load("model.onnx")

# 🔹 Convert ONNX model to PyTorch
pytorch_model = ConvertModel(onnx_model)

# Test the PyTorch model
dummy_input = torch.randn(1, 4)
output = pytorch_model(dummy_input)

print("PyTorch model output:", output)
