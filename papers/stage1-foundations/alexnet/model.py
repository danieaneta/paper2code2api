"""AlexNet architecture (adapted for CIFAR-10).

Reference: Krizhevsky, Sutskever, Hinton (2012),
"ImageNet Classification with Deep Convolutional Neural Networks", NeurIPS.

This is a faithful-but-modernized reproduction, scaled for laptop training:
  * The original took 3x224x224 ImageNet images (1000 classes) and trained for a
    week on two GPUs. We take 3x32x32 CIFAR-10 images (10 classes) so the whole
    lesson trains from scratch on a CPU in minutes. Kernel/stride sizes are scaled
    to the smaller input -- but we keep AlexNet's habit of downsampling early
    (its first conv used stride 4); ours uses stride 2 to shrink the map up front,
    which is what makes it fast. See README.md.
  * We keep AlexNet's defining ideas exactly: ReLU activations, five convolutional
    layers (96 -> 256 -> 384 -> 384 -> 256 channels), max pooling, and two large
    dropout-regularized fully-connected layers before the classifier.
  * The original used Local Response Normalization (LRN). We use BatchNorm, the
    modern standard that replaced it -- it also makes CPU training converge fast.
    See the "Faithful vs modernized" section of README.md.
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet for 3x32x32 color input (CIFAR-10), 10-class output."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # conv1: 3x32x32 -> 96x16x16 (stride-2 downsample), then pool -> 96x8x8
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv2: -> 256x8x8, then pool -> 256x4x4
            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3: -> 384x4x4
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv4: -> 384x4x4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # conv5: -> 256x4x4, then pool -> 256x2x2
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 256x2x2 -> 1024
            nn.Dropout(0.5),
            nn.Linear(256 * 2 * 2, 4096),     # FC6
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),            # FC7
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),     # FC8 (output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    # Sanity check: shapes flow end to end.
    model = AlexNet()
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(out.shape)}  |  parameters: {n_params:,}")
