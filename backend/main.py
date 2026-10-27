import io
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "mobilenet": BASE_DIR / "models" / "best_mobilenetv2.keras",
    "scratch": BASE_DIR / "models" / "bread_scratch_cnn.keras",
}
app = FastAPI(title="BreadGuard AI Prediction API", version="1.0.0")
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])
_models = {}


def get_model(name: str):
    if name not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    if name not in _models:
        path = MODEL_PATHS[name]
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Model file missing: {path.name}")
        _models[name] = tf.keras.models.load_model(path, compile=False)
    return _models[name]


@app.get("/health")
def health():
    return {"status": "ok", "models_available": {key: path.exists() for key, path in MODEL_PATHS.items()}}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("mobilenet")):
    if model not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unsupported model")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG or WEBP image")
    content = await file.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB")
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image = image.resize((224, 224))
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file")
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    # Both saved models include their own preprocessing. Do not rescale again.
    try:
        probability = float(get_model(model).predict(batch, verbose=0)[0][0])
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Model inference failed") from exc
    if not np.isfinite(probability):
        raise HTTPException(status_code=500, detail="Model returned an invalid probability")
    probability = max(0.0, min(1.0, probability))
    return {
        "model": model,
        "prediction": "spoiled" if probability >= 0.5 else "fresh",
        "spoiled_probability": probability,
        "fresh_probability": 1 - probability,
        "confidence": max(probability, 1 - probability),
        "disclaimer": "Image-based visible spoilage screening only; not a food safety guarantee.",
    }
