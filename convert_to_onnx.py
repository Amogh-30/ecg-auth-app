import tensorflow as tf
import tf2onnx
import os

print("--- Starting Keras to ONNX Conversion ---")

# Define file paths
keras_model_path = "feature_model_all.keras"
onnx_model_path = "feature_model.onnx"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# 1. Load your trained Keras model
print(f"Loading Keras model from {keras_model_path}...")
try:
    model = tf.keras.models.load_model(keras_model_path)
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit()

print("Model loaded successfully.")

# 2. Define the "input signature"
# This tells ONNX what kind of data to expect.
# Your model expects: (batch_size, 1500, 1)
spec = (tf.TensorSpec((None, 1500, 1), tf.float32, name="input"),)

# 3. Convert the model
print("Converting to ONNX format...")
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=spec, 
    opset=13, # A standard version
    output_path=onnx_model_path
)

print(f"--- Conversion Complete! ---")
print(f"Model saved to: {onnx_model_path}")