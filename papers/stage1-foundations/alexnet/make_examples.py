"""Generate input/output example images for the README.

Produces (into ./assets):
  * examples_grid.png      - a grid of input photos with predicted label + confidence,
                             green border = correct, red = wrong.
  * prediction_detail.png  - a single input photo next to its full output probability bar chart.

Usage:
    python make_examples.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from torchvision import datasets

from infer import load_model, preprocess, CLASSES

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

N_GRID = 10  # number of photos in the grid


def main() -> None:
    model = load_model()
    test_ds = datasets.CIFAR10(HERE / "data", train=False, download=True)

    # ---- 1) grid of inputs with predicted outputs ----
    cols = 5
    rows = (N_GRID + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.4))
    fig.suptitle("AlexNet — inputs (CIFAR-10) and predictions", fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        img, true_label = test_ds[i]
        with torch.no_grad():
            probs = torch.softmax(model(preprocess(img)), dim=1).squeeze(0)
        pred = int(probs.argmax())
        conf = float(probs[pred])
        correct = pred == true_label
        color = "#2e7d32" if correct else "#c62828"

        ax.imshow(img)
        ax.set_title(f"{CLASSES[pred]}  ({conf*100:.0f}%)\ntrue: {CLASSES[true_label]}",
                     color=color, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2.5)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.55)
    fig.savefig(ASSETS / "examples_grid.png", dpi=130)
    plt.close(fig)

    # ---- 2) single input -> output probability distribution ----
    img, true_label = test_ds[0]
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(img)), dim=1).squeeze(0).tolist()
    pred = int(max(range(10), key=lambda k: probs[k]))

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 3.6),
                                         gridspec_kw={"width_ratios": [1, 1.8]})
    ax_img.imshow(img)
    ax_img.set_title("INPUT\n(32x32 color)", fontsize=10, fontweight="bold")
    ax_img.set_xticks([]); ax_img.set_yticks([])

    bars = ax_bar.bar(range(10), probs, color="#bdbdbd")
    bars[pred].set_color("#2e7d32")
    ax_bar.set_title(f"OUTPUT\nprediction = {CLASSES[pred]}  ({probs[pred]*100:.1f}%)",
                     fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("probability")
    ax_bar.set_xticks(range(10))
    ax_bar.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(ASSETS / "prediction_detail.png", dpi=130)
    plt.close(fig)

    print(f"wrote {ASSETS/'examples_grid.png'}")
    print(f"wrote {ASSETS/'prediction_detail.png'}")


if __name__ == "__main__":
    main()
