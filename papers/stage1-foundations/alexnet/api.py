"""FastAPI inference server for AlexNet (CIFAR-10).

Implements the shared paper2code2api contract:
    POST /predict   (multipart image file)  -> JSON {prediction, confidence, probabilities}
    GET  /health    -> {status, model_loaded}

Run:
    pip install -r requirements.txt
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError

from infer import predict, load_model, WEIGHTS_PATH, CLASSES

app = FastAPI(
    title="paper2code2api · AlexNet",
    description="Image classification (CIFAR-10) from Krizhevsky et al. 2012.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": WEIGHTS_PATH.exists()}


@app.get("/classes")
def classes() -> dict:
    return {"classes": CLASSES}


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(..., description="Image to classify into a CIFAR-10 category"),
) -> dict:
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    try:
        return predict(img)
    except FileNotFoundError as exc:
        # Model not trained yet.
        raise HTTPException(status_code=503, detail=str(exc))


@app.on_event("startup")
def _warm() -> None:
    # Load weights eagerly so the first request isn't slow; tolerate untrained state.
    try:
        load_model()
    except FileNotFoundError:
        pass
