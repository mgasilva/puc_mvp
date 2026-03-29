"""
Pneumonia Detection API
FastAPI backend for chest X-ray classification using EfficientNet-B0
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pneumonia Detection API",
    description="Chest X-ray classification: NORMAL vs PNEUMONIA using EfficientNet-B0",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "best_pneumonia_model.pth")
CLASSES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path: str) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# Load once at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at '{MODEL_PATH}'. "
        "Copy best_pneumonia_model.pth next to app.py and restart."
    )

model = load_model(MODEL_PATH)
print(f"✅ Model loaded from '{MODEL_PATH}' on {DEVICE}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = eval_transforms(img).unsqueeze(0)   # add batch dim
    return tensor.to(DEVICE)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Pneumonia Detection API is running."}

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model": "EfficientNet-B0",
        "classes": CLASSES,
    }

@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image (JPEG or PNG) and receive a prediction.

    Returns:
    - **prediction**: NORMAL or PNEUMONIA
    - **confidence**: probability of the predicted class (0–1)
    - **probabilities**: full softmax distribution
    """
    # Validate content type
    allowed = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG or PNG.",
        )

    image_bytes = await file.read()

    try:
        tensor = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()

    pred_idx = int(torch.argmax(torch.tensor(probs)))
    prediction = CLASSES[pred_idx]
    confidence = probs[pred_idx]

    return JSONResponse({
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {
            "NORMAL":    round(probs[0], 4),
            "PNEUMONIA": round(probs[1], 4),
        },
        "filename": file.filename,
    })
