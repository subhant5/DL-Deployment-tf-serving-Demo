import tensorflow as tf
import os

# Create directory for the model
model_dir = "mobilenet_v2/1"  # TF Serving expects models to be in a numbered subdirectory
os.makedirs(model_dir, exist_ok=True)

# Load MobileNetV2 model
model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=True,
    weights='imagenet'
)

# Save the model in SavedModel format
model.export(model_dir)
print(f"Model saved to {model_dir}")