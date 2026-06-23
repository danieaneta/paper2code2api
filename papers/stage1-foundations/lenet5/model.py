"""LeNet-5 architecture.

Reference: LeCun, Bottou, Bengio, Haffner (1998),
"Gradient-Based Learning Applied to Document Recognition", Proc. IEEE.

This is a faithful-but-modernized reproduction:
  * Input is 1x32x32 (MNIST 28x28 is padded by 2 on each side, as in the paper).
  * Activations are tanh and subsampling is average pooling, matching the 1998 design.
  * The original C3 used a sparse, hand-designed connection table and the output
    layer used Gaussian (RBF) units. We use full connections and a standard linear
    output + cross-entropy, which is the common modern reproduction. See README.md.
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """Classic LeNet-5 for 32x32 grayscale input, 10-class output."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),   # C1: 1x32x32 -> 6x28x28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S2: -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5),  # C3: -> 16x10x10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S4: -> 16x5x5
            nn.Conv2d(16, 120, kernel_size=5),  # C5: -> 120x1x1
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84),  # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),  # output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    # Sanity check: shapes flow end to end.
    model = LeNet5()
    dummy = torch.randn(1, 1, 32, 32)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(out.shape)}  |  parameters: {n_params:,}")
