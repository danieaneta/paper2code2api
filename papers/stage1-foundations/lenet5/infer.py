"""Inference utilities for LeNet-5.

Shared by api.py and usable standalone:
    python infer.py path/to/digit.png
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
from functools import lru_cache

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from model import LeNet5

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "lenet5.pt"

# Same normalization/padding as training, plus grayscale + resize so arbitrary
# uploaded images are coerced to the 28x28 MNIST grid before the 32x32 pad.
_PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Pad(2),
])


@lru_cache(maxsize=1)
def load_model() -> LeNet5:
    """Load weights once and cache. Raises if the model has not been trained."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"weights not found at {WEIGHTS_PATH}. Run `python train.py` first."
        )
    model = LeNet5()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


def preprocess(img: Image.Image, invert: bool = False) -> torch.Tensor:
    """PIL image -> 1x1x32x32 tensor. MNIST is white-on-black; set invert=True
    for typical black-on-white digit photos/drawings."""
    if invert:
        img = ImageOps.invert(img.convert("L"))
    return _PREPROCESS(img).unsqueeze(0)


@torch.no_grad()
def predict(img: Image.Image, invert: bool = False) -> dict:
    """Return prediction, confidence, and full probability vector."""
    model = load_model()
    logits = model(preprocess(img, invert=invert))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred = int(probs.argmax().item())
    return {
        "prediction": pred,
        "confidence": float(probs[pred].item()),
        "probabilities": [round(float(p), 6) for p in probs.tolist()],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python infer.py <image_path> [--invert]")
        raise SystemExit(1)
    image = Image.open(sys.argv[1])
    result = predict(image, invert="--invert" in sys.argv)
    print(result)
