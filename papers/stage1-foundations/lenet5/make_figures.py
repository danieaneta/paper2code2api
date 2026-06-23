"""Generate teaching figures for the lesson README (into ./assets).

Produces:
  * architecture.png   - the LeNet-5 layer-by-layer block diagram.
  * convolution.png    - a filter sliding over the input to make a feature map.
  * pooling.png        - 2x2 average pooling shrinking a feature map.
  * feature_maps.png   - REAL C1 and C3 feature maps from the trained model on a digit.

Usage:
    python make_figures.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec
from torchvision import datasets

from infer import load_model, preprocess

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
ASSETS.mkdir(exist_ok=True)

CONV = "#1565c0"   # blue
POOL = "#2e7d32"   # green
FC = "#c62828"     # red
INPUT = "#616161"  # gray


def fig_architecture() -> None:
    layers = [
        ("INPUT", "1×32×32", INPUT),
        ("C1 conv 5×5", "6×28×28", CONV),
        ("S2 pool 2×2", "6×14×14", POOL),
        ("C3 conv 5×5", "16×10×10", CONV),
        ("S4 pool 2×2", "16×5×5", POOL),
        ("C5 conv 5×5", "120", CONV),
        ("F6 dense", "84", FC),
        ("OUTPUT", "10 digits", FC),
    ]
    fig, ax = plt.subplots(figsize=(14, 3.2))
    box_w, gap, x, y, box_h = 1.4, 0.45, 0.0, 0.0, 1.2
    for i, (name, shape, color) in enumerate(layers):
        ax.add_patch(Rectangle((x, y), box_w, box_h, facecolor=color, alpha=0.85,
                               edgecolor="black", linewidth=1.2))
        ax.text(x + box_w / 2, y + box_h * 0.62, name, ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        ax.text(x + box_w / 2, y + box_h * 0.28, shape, ha="center", va="center",
                color="white", fontsize=8.5)
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((x + box_w, y + box_h / 2),
                                         (x + box_w + gap, y + box_h / 2),
                                         arrowstyle="-|>", mutation_scale=14, color="black"))
        x += box_w + gap

    ax.text(x / 2, box_h + 0.55, "LeNet-5 architecture — data flows left to right",
            ha="center", fontsize=13, fontweight="bold")
    # legend
    for j, (lbl, c) in enumerate([("Convolution", CONV), ("Pooling", POOL), ("Dense / output", FC)]):
        ax.add_patch(Rectangle((j * 3.4, -0.9), 0.4, 0.3, facecolor=c, alpha=0.85))
        ax.text(j * 3.4 + 0.55, -0.75, lbl, va="center", fontsize=9)

    ax.set_xlim(-0.3, x); ax.set_ylim(-1.1, box_h + 0.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "architecture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_convolution(patch: np.ndarray) -> None:
    """patch: an 8x8 grayscale crop of a real digit."""
    fig, (ax_in, ax_mid, ax_out) = plt.subplots(
        1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 0.5, 1.0]})

    ax_in.imshow(patch, cmap="gray")
    ax_in.add_patch(Rectangle((-0.5, -0.5), 5, 5, fill=False, edgecolor="#ff1744", linewidth=3))
    ax_in.add_patch(Rectangle((0.5, 0.5), 5, 5, fill=False, edgecolor="#ff1744",
                              linewidth=1.5, linestyle="--"))
    ax_in.set_title("INPUT\nthe 5×5 filter (red) slides\nacross every position",
                    fontsize=10, fontweight="bold")
    ax_in.set_xticks([]); ax_in.set_yticks([])

    ax_mid.axis("off")
    ax_mid.text(0.5, 0.62, "filter\n(5×5 learned\nweights)\n+ bias", ha="center", va="center",
                fontsize=10, bbox=dict(boxstyle="round", facecolor=CONV, alpha=0.2))
    ax_mid.annotate("", xy=(1.05, 0.5), xytext=(-0.05, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
    ax_mid.text(0.5, 0.30, "multiply\n+ sum", ha="center", va="center", fontsize=9)

    out = np.zeros((4, 4))
    out[0, 0] = 1
    ax_out.imshow(out, cmap="Greens", vmin=0, vmax=1)
    ax_out.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#ff1744", linewidth=3))
    ax_out.set_title("FEATURE MAP\neach filter position →\none output value", fontsize=10, fontweight="bold")
    ax_out.set_xticks([]); ax_out.set_yticks([])

    fig.suptitle("Convolution: one shared filter sweeps the image, building a feature map",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ASSETS / "convolution.png", dpi=130)
    plt.close(fig)


def fig_pooling() -> None:
    grid = np.array([[3, 8, 1, 0],
                     [5, 6, 2, 4],
                     [7, 2, 9, 3],
                     [1, 0, 5, 1]], dtype=float)
    pooled = grid.reshape(2, 2, 2, 2).mean(axis=(1, 3))  # 2x2 averages

    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 0.4, 1.0]})

    axL.imshow(np.ones_like(grid), cmap="Greys", vmin=0, vmax=1)
    block_colors = ["#ffcdd2", "#c8e6c9", "#bbdefb", "#fff9c4"]
    for bi, (r, c) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        axL.add_patch(Rectangle((c - 0.5, r - 0.5), 2, 2, facecolor=block_colors[bi],
                                edgecolor="black", linewidth=2))
    for r in range(4):
        for c in range(4):
            axL.text(c, r, int(grid[r, c]), ha="center", va="center", fontsize=12)
    axL.set_title("FEATURE MAP (4×4)\nsplit into 2×2 blocks", fontsize=10, fontweight="bold")
    axL.set_xlim(-0.5, 3.5); axL.set_ylim(3.5, -0.5)
    axL.set_xticks([]); axL.set_yticks([])

    axM.axis("off")
    axM.annotate("", xy=(1.0, 0.5), xytext=(0.0, 0.5),
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="black"))
    axM.text(0.5, 0.62, "2×2 average\npool", ha="center", va="center", fontsize=10)

    axR.imshow(np.ones_like(pooled), cmap="Greys", vmin=0, vmax=1)
    for bi, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        axR.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=block_colors[bi],
                                edgecolor="black", linewidth=2))
    for r in range(2):
        for c in range(2):
            axR.text(c, r, f"{pooled[r, c]:.1f}", ha="center", va="center", fontsize=13, fontweight="bold")
    axR.set_title("POOLED (2×2)\neach cell = block average", fontsize=10, fontweight="bold")
    axR.set_xlim(-0.5, 1.5); axR.set_ylim(1.5, -0.5)
    axR.set_xticks([]); axR.set_yticks([])

    fig.suptitle("Average pooling: summarize each 2×2 block → smaller, shift-tolerant map",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ASSETS / "pooling.png", dpi=130)
    plt.close(fig)


def fig_feature_maps(img) -> None:
    """Run a real digit through the trained model and show C1 and C3 activations."""
    model = load_model()
    x = preprocess(img)
    acts = {}
    h = x
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            h = layer(h)
            if i == 1:   # after C1 + tanh
                acts["C1"] = h.squeeze(0).numpy()   # 6x28x28
            if i == 4:   # after C3 + tanh
                acts["C3"] = h.squeeze(0).numpy()   # 16x10x10

    fig = plt.figure(figsize=(11, 6.2))
    outer = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1.0, 2.0], hspace=0.55)

    # input (centered)
    gs_in = gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=outer[0])
    ax_in = fig.add_subplot(gs_in[0, 4])
    ax_in.imshow(img, cmap="gray"); ax_in.set_title("INPUT digit", fontsize=10, fontweight="bold")
    ax_in.set_xticks([]); ax_in.set_yticks([])

    # C1: 6 maps
    gs_c1 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[1], wspace=0.15)
    for i in range(6):
        ax = fig.add_subplot(gs_c1[0, i])
        ax.imshow(acts["C1"][i], cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("C1\n6 maps", fontsize=9, fontweight="bold", rotation=0, labelpad=22, va="center")

    # C3: 16 maps (2x8)
    gs_c3 = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[2], wspace=0.15, hspace=0.15)
    for i in range(16):
        ax = fig.add_subplot(gs_c3[i // 8, i % 8])
        ax.imshow(acts["C3"][i], cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("C3\n16 maps", fontsize=9, fontweight="bold", rotation=0, labelpad=22, va="center")

    fig.suptitle("What the trained network actually sees — real feature maps\n"
                 "early layers (C1) catch edges/strokes; deeper layers (C3) catch combinations",
                 fontsize=12, fontweight="bold")
    fig.savefig(ASSETS / "feature_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    test_ds = datasets.MNIST(HERE / "data", train=False, download=True)
    img, _ = test_ds[0]  # a handwritten 7
    patch = np.array(img.resize((8, 8)), dtype=float)  # small crop for the conv diagram

    fig_architecture()
    fig_convolution(patch)
    fig_pooling()
    fig_feature_maps(img)
    for name in ("architecture", "convolution", "pooling", "feature_maps"):
        print(f"wrote {ASSETS / (name + '.png')}")


if __name__ == "__main__":
    main()
