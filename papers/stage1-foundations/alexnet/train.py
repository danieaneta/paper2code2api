"""Train AlexNet on CIFAR-10.

Usage:
    python train.py                 # 15 epochs, saves alexnet.pt
    python train.py --epochs 20     # custom epochs

Downloads CIFAR-10 to ./data on first run (~170 MB).
"""

# Must be set before torch imports OpenMP-linked libs (Windows duplicate-libiomp fix).
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import AlexNet

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "alexnet.pt"

# CIFAR-10's per-channel mean/std (computed over the training set).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Training uses AlexNet's own data augmentation -- random crops and horizontal
# flips -- to multiply the effective dataset and fight overfitting.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Evaluation/inference: no augmentation, just normalize. (Re-used by infer.py.)
EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def train(epochs: int = 15, batch_size: int = 128, lr: float = 1e-3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = datasets.CIFAR10(HERE / "data", train=True, download=True, transform=TRAIN_TRANSFORM)
    test_ds = datasets.CIFAR10(HERE / "data", train=False, download=True, transform=EVAL_TRANSFORM)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1000)

    model = AlexNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for images, labels in train_dl:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_ds)

        acc = evaluate(model, test_dl, device)
        print(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  test_acc={acc:.4f}")

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"saved weights -> {WEIGHTS_PATH}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
