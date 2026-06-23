"""Train LeNet-5 on MNIST.

Usage:
    python train.py                 # 5 epochs, saves lenet5.pt
    python train.py --epochs 10     # custom epochs

Downloads MNIST to ./data on first run.
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

from model import LeNet5

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "lenet5.pt"

# MNIST is 28x28; pad to 32x32 for the original LeNet receptive field.
# 0.1307 / 0.3081 are the standard MNIST mean/std.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Pad(2),
])


def train(epochs: int = 5, batch_size: int = 64, lr: float = 1e-3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = datasets.MNIST(HERE / "data", train=True, download=True, transform=TRANSFORM)
    test_ds = datasets.MNIST(HERE / "data", train=False, download=True, transform=TRANSFORM)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1000)

    model = LeNet5().to(device)
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
    parser = argparse.ArgumentParser(description="Train LeNet-5 on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
