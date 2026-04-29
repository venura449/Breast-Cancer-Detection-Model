import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

model = tf.keras.models.load_model("breast_histopathology_cnn_final.keras")

app = FastAPI(title="Breast Histopathology API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (50, 50))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.get("/")
def home():
    return {"message": "Breast Histopathology CNN API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)
        prob = float(model.predict(img)[0][0])
        label = "IDC Positive / Cancer" if prob >= 0.5 else "IDC Negative / Non-cancer"
        return {
            "probability": prob,
            "prediction": label,
            "class": 1 if prob >= 0.5 else 0
        }
    except Exception as e:
        return {"error": str(e)}