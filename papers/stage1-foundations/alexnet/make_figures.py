"""Generate teaching figures for the lesson README (into ./assets).

Produces:
  * architecture.png   - the AlexNet layer-by-layer block diagram (CIFAR-10 variant).
  * relu.png           - ReLU vs tanh: why the activation swap speeds up training.
  * dropout.png        - dropout switching off half the neurons on a training step.
  * feature_maps.png   - REAL conv1 and conv5 feature maps from the trained model.

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
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
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
        ("INPUT", "3×32×32", INPUT),
        ("conv1 3×3 /2", "96×16×16", CONV),
        ("pool 2×2", "96×8×8", POOL),
        ("conv2 3×3", "256×8×8", CONV),
        ("pool 2×2", "256×4×4", POOL),
        ("conv3 3×3", "384×4×4", CONV),
        ("conv4 3×3", "384×4×4", CONV),
        ("conv5 3×3", "256×4×4", CONV),
        ("pool 2×2", "256×2×2", POOL),
        ("FC6 dense", "4096", FC),
        ("FC7 dense", "4096", FC),
        ("OUTPUT", "10 classes", FC),
    ]
    fig, ax = plt.subplots(figsize=(17, 3.2))
    box_w, gap, x, y, box_h = 1.15, 0.32, 0.0, 0.0, 1.2
    for i, (name, shape, color) in enumerate(layers):
        ax.add_patch(Rectangle((x, y), box_w, box_h, facecolor=color, alpha=0.85,
                               edgecolor="black", linewidth=1.2))
        ax.text(x + box_w / 2, y + box_h * 0.62, name, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        ax.text(x + box_w / 2, y + box_h * 0.28, shape, ha="center", va="center",
                color="white", fontsize=7.5)
        if i < len(layers) - 1:
            ax.add_patch(FancyArrowPatch((x + box_w, y + box_h / 2),
                                         (x + box_w + gap, y + box_h / 2),
                                         arrowstyle="-|>", mutation_scale=12, color="black"))
        x += box_w + gap

    ax.text(x / 2, box_h + 0.55, "AlexNet architecture (CIFAR-10 variant) — data flows left to right",
            ha="center", fontsize=13, fontweight="bold")
    for j, (lbl, c) in enumerate([("Convolution", CONV), ("Pooling", POOL), ("Dense / output", FC)]):
        ax.add_patch(Rectangle((j * 4.0, -0.9), 0.4, 0.3, facecolor=c, alpha=0.85))
        ax.text(j * 4.0 + 0.55, -0.75, lbl, va="center", fontsize=9)

    ax.set_xlim(-0.3, x); ax.set_ylim(-1.1, box_h + 0.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "architecture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_relu() -> None:
    x = np.linspace(-4, 4, 400)
    relu = np.maximum(0, x)
    tanh = np.tanh(x)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))

    axL.plot(x, tanh, color="#c62828", linewidth=2.5)
    axL.axhline(1, color="gray", ls="--", lw=1); axL.axhline(-1, color="gray", ls="--", lw=1)
    axL.set_title("tanh  (LeNet, 1998)\nsaturates — gradient → 0 at the edges",
                  fontsize=10, fontweight="bold")
    axL.annotate("flat here:\nlearning stalls", xy=(3, 0.995), xytext=(0.2, 0.45),
                 fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black"))

    axR.plot(x, relu, color="#1565c0", linewidth=2.5)
    axR.set_title("ReLU  f(x) = max(0, x)  (AlexNet, 2012)\nno saturation for x>0 — fast, stable gradients",
                  fontsize=10, fontweight="bold")
    axR.annotate("slope stays 1:\nlearning keeps flowing", xy=(3, 3), xytext=(-3.8, 2.4),
                 fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black"))

    for ax in (axL, axR):
        ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("input"); ax.set_ylabel("output")
        ax.grid(alpha=0.25)
    axL.set_ylim(-1.5, 4); axR.set_ylim(-1.5, 4)

    fig.suptitle("AlexNet's first big change: swap tanh for ReLU", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ASSETS / "relu.png", dpi=130)
    plt.close(fig)


def _draw_net(ax, dropped: set, title: str) -> None:
    """Draw a tiny 3-layer net; hidden units in `dropped` are switched off."""
    in_y = [3.5, 2.5, 1.5, 0.5]
    hid_y = [4.0, 3.2, 2.4, 1.6, 0.8, 0.0]
    out_y = [3.0, 2.0, 1.0]
    cols = {"in": 0.0, "hid": 1.6, "out": 3.2}

    # connections
    for hy_i, hy in enumerate(hid_y):
        if hy_i in dropped:
            continue
        for iy in in_y:
            ax.plot([cols["in"], cols["hid"]], [iy, hy], color="#cfd8dc", lw=0.8, zorder=1)
        for oy in out_y:
            ax.plot([cols["hid"], cols["out"]], [hy, oy], color="#cfd8dc", lw=0.8, zorder=1)

    # nodes
    for iy in in_y:
        ax.add_patch(Circle((cols["in"], iy), 0.16, color="#90a4ae", zorder=2))
    for hy_i, hy in enumerate(hid_y):
        if hy_i in dropped:
            ax.add_patch(Circle((cols["hid"], hy), 0.16, facecolor="white",
                                edgecolor="#b0bec5", zorder=2))
            ax.plot([cols["hid"] - 0.13, cols["hid"] + 0.13], [hy - 0.13, hy + 0.13],
                    color="#c62828", lw=2, zorder=3)
            ax.plot([cols["hid"] - 0.13, cols["hid"] + 0.13], [hy + 0.13, hy - 0.13],
                    color="#c62828", lw=2, zorder=3)
        else:
            ax.add_patch(Circle((cols["hid"], hy), 0.16, color="#1565c0", zorder=2))
    for oy in out_y:
        ax.add_patch(Circle((cols["out"], oy), 0.16, color="#c62828", zorder=2))

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, 3.7); ax.set_ylim(-0.5, 4.6)
    ax.axis("off")


def fig_dropout() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    _draw_net(axL, dropped=set(), title="Full network\nevery neuron active")
    _draw_net(axR, dropped={1, 3, 4}, title="With dropout (p=0.5)\nhalf the hidden units switched off this step")
    fig.suptitle("Dropout: each training step randomly drops neurons, so none can rely on\n"
                 "any single partner — a cheap, powerful cure for overfitting",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(ASSETS / "dropout.png", dpi=130)
    plt.close(fig)


def fig_feature_maps(img) -> None:
    """Run a real photo through the trained model and show conv1 and conv5 activations."""
    model = load_model()
    x = preprocess(img)
    acts = {}
    h = x
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            h = layer(h)
            if i == 2:    # after conv1 + BN + ReLU
                acts["conv1"] = h.squeeze(0).numpy()   # 96x32x32
            if i == 16:   # after conv5 + BN + ReLU
                acts["conv5"] = h.squeeze(0).numpy()   # 256x8x8

    fig = plt.figure(figsize=(11, 6.4))
    outer = gridspec.GridSpec(3, 1, height_ratios=[1.5, 1.0, 2.0], hspace=0.55)

    gs_in = gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=outer[0])
    ax_in = fig.add_subplot(gs_in[0, 4])
    ax_in.imshow(img); ax_in.set_title("INPUT photo", fontsize=10, fontweight="bold")
    ax_in.set_xticks([]); ax_in.set_yticks([])

    gs_c1 = gridspec.GridSpecFromSubplotSpec(1, 8, subplot_spec=outer[1], wspace=0.15)
    for i in range(8):
        ax = fig.add_subplot(gs_c1[0, i])
        ax.imshow(acts["conv1"][i], cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("conv1\n8 of 96", fontsize=9, fontweight="bold",
                          rotation=0, labelpad=24, va="center")

    gs_c5 = gridspec.GridSpecFromSubplotSpec(2, 8, subplot_spec=outer[2], wspace=0.15, hspace=0.15)
    for i in range(16):
        ax = fig.add_subplot(gs_c5[i // 8, i % 8])
        ax.imshow(acts["conv5"][i], cmap="viridis"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("conv5\n16 of 256", fontsize=9, fontweight="bold",
                          rotation=0, labelpad=24, va="center")

    fig.suptitle("What the trained network actually sees — real feature maps\n"
                 "early layers (conv1) catch edges/colors; deep layers (conv5) catch abstract parts",
                 fontsize=12, fontweight="bold")
    fig.savefig(ASSETS / "feature_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    test_ds = datasets.CIFAR10(HERE / "data", train=False, download=True)
    img, _ = test_ds[0]  # a real CIFAR-10 photo

    fig_architecture()
    fig_relu()
    fig_dropout()
    fig_feature_maps(img)
    for name in ("architecture", "relu", "dropout", "feature_maps"):
        print(f"wrote {ASSETS / (name + '.png')}")


if __name__ == "__main__":
    main()
