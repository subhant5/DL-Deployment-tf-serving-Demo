from fastapi import FastAPI, HTTPException, UploadFile, File
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import json

app = FastAPI()

# Load ImageNet class names from local file
IMAGENET_CLASSES = {}
try:
    with open('imagenet_classes.txt', 'r') as f:
        IMAGENET_CLASSES = {i: name.strip() for i, name in enumerate(f.readlines())}
except Exception as e:
    print(f"Error loading ImageNet classes: {str(e)}")

async def preprocess_image(image_data: bytes):
    try:
        # Open image from bytes
        img = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224
        img = img.resize((224, 224), Image.BILINEAR)
        
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        
        # Scale pixels to [0, 1]
        img_array = img_array / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Convert to list for JSON serialization
        return img_array.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image_data = await file.read()
        
        # Preprocess the image
        processed_image = await preprocess_image(image_data)
        
        # Prepare request for TensorFlow Serving
        payload = {
            "signature_name": "serving_default",
            "instances": processed_image
        }
        
        # Call TensorFlow Serving
        model_url = 'http://tensorflow_serving:8501/v1/models/mobilenet_v2:predict'
        response = requests.post(
            model_url,
            json=payload,
            headers={"content-type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"Error response from model: {response.text}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {response.text}")
        
        # Get predictions and decode class names
        predictions = response.json()['predictions'][0]
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]
        
        # Get actual class name from ImageNet classes
        class_name = IMAGENET_CLASSES.get(class_idx, f"Unknown class {class_idx}")
        
        return {
            "class_id": int(class_idx),
            "class_name": class_name,
            "confidence": float(confidence)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")