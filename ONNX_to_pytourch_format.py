# Step 1: Convert TensorFlow Model to ONNX

import tensorflow as tf
import tf2onnx

# 1️⃣ Build a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 2️⃣ Save the model as a TensorFlow SavedModel
model.save("tf_model")

# 3️⃣ Convert the model to ONNX format
!python -m tf2onnx.convert --saved-model tf_model --output model.onnx

# Step 2: Import the ONNX Model into PyTorch

import torch
from onnx2pytorch import ConvertModel

# 1️⃣ Load the ONNX model
onnx_model_path = "model.onnx"

# 2️⃣ Convert it to a PyTorch model
pytorch_model = ConvertModel(onnx_model_path)

# 3️⃣ Test the model
dummy_input = torch.randn(1, 4)
output = pytorch_model(dummy_input)
print("PyTorch model output:", output)

