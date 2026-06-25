"""Inference utilities for AlexNet (CIFAR-10).

Shared by api.py and usable standalone:
    python infer.py path/to/photo.png
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
from functools import lru_cache

import torch
from PIL import Image
from torchvision import transforms

from model import AlexNet

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "alexnet.pt"

# The 10 CIFAR-10 classes, in the dataset's label order (index = class id).
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Same normalization as training. Arbitrary uploads are coerced to RGB and to the
# 32x32 CIFAR grid first. These constants MUST match train.py.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

_PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


@lru_cache(maxsize=1)
def load_model() -> AlexNet:
    """Load weights once and cache. Raises if the model has not been trained."""
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"weights not found at {WEIGHTS_PATH}. Run `python train.py` first."
        )
    model = AlexNet()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    """PIL image -> 1x3x32x32 tensor (forced to RGB, resized to the CIFAR grid)."""
    return _PREPROCESS(img.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict(img: Image.Image) -> dict:
    """Return predicted class name, confidence, and full probability vector."""
    model = load_model()
    logits = model(preprocess(img))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    idx = int(probs.argmax().item())
    return {
        "prediction": CLASSES[idx],
        "confidence": float(probs[idx].item()),
        "probabilities": {CLASSES[i]: round(float(p), 6) for i, p in enumerate(probs.tolist())},
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python infer.py <image_path>")
        raise SystemExit(1)
    image = Image.open(sys.argv[1])
    result = predict(image)
    print(result)
