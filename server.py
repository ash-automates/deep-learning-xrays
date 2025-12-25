"""
FastAPI app serving chest X-ray classification predictions and a simple upload UI.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from model import ChestXRayClassifier
from preprocessing import ChestXRayPreprocessor

ENV_CLASSES = os.getenv("CLASSES")
CLASSES: List[str] = (
    [c.strip() for c in ENV_CLASSES.split(",") if c.strip()]
    if ENV_CLASSES
    else ["healthy", "viral_pneumonia", "bacterial_pneumonia", "covid19"]
)

MODEL_NAME = os.getenv("MODEL_NAME", "resnet50")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "imagenet")
MODEL_PATH = os.getenv("MODEL_PATH", f"{MODEL_NAME}_chest_xray_classifier.h5")

app = FastAPI(title="Chest X-Ray Classifier", version="1.0.0")
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

preprocessor = ChestXRayPreprocessor(target_size=(224, 224))
_classifier: ChestXRayClassifier | None = None


def _load_classifier() -> ChestXRayClassifier:
    """Load a saved model if present; otherwise build a fresh model head."""
    weights = MODEL_WEIGHTS if MODEL_WEIGHTS.lower() != "none" else None
    classifier = ChestXRayClassifier(
        model_name=MODEL_NAME,
        num_classes=len(CLASSES),
        input_shape=(224, 224, 3),
        weights=weights,
    )

    if os.path.exists(MODEL_PATH):
        classifier.load_model(MODEL_PATH)
    else:
        classifier.build_model(freeze_base=True)
    return classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier
    _classifier = _load_classifier()
    yield


app.router.lifespan_context = lifespan


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index_path = Path("frontend/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "ok", "message": "Frontend not found"})


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model_loaded": str(_classifier is not None).lower()}


def _is_plausible_xray(color_image: np.ndarray) -> bool:
    # Compute colorfulness metric; X-rays are near grayscale
    (b, g, r) = cv2.split(color_image.astype("float"))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_root = np.sqrt(np.mean(rg**2) + np.mean(yb**2))
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    colorfulness = std_root + (0.3 * mean_root)

    # Edge density: avoid blank or purely synthetic images
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size

    # Aspect ratio sanity check
    h, w, _ = color_image.shape
    aspect_ok = 0.4 <= (w / h) <= 2.5

    return colorfulness < 18.0 and edge_density > 0.005 and aspect_ok


def _preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    color_image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if color_image is None:
        raise ValueError("Unable to decode image")

    if not _is_plausible_xray(color_image):
        raise ValueError("Image does not appear to be a chest X-ray")

    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    resized = preprocessor.resize_image(gray)
    image = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    image = preprocessor.normalize_image(image)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    if _classifier is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/bmp"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        processed = _preprocess_bytes(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    preds = _classifier.predict(np.expand_dims(processed, axis=0))
    top_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    probabilities = {CLASSES[i]: float(preds[0][i]) for i in range(len(CLASSES))}

    return {
        "predicted_class": CLASSES[top_idx],
        "confidence": confidence,
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
