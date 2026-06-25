# License notes — AlexNet

| Layer | Status |
|---|---|
| **Paper** | Krizhevsky, Sutskever, Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks* — academic paper, freely available. |
| **Architecture** | AlexNet is a foundational architecture with no patent/license encumbrance for reimplementation. This implementation is original code written for paper2code2api (adapted for CIFAR-10 — see `model.py` / `README.md`). |
| **Weights** | Trained from scratch here on CIFAR-10 → produced by this repo, no upstream weight license to inherit. (Note: torchvision also ships ImageNet-pretrained AlexNet weights under its BSD-3-Clause license; we do **not** use those — our weights are our own.) |
| **Dataset (CIFAR-10)** | Collected by Krizhevsky, Nair & Hinton (a labeled subset of the original 80 Million Tiny Images). Freely distributed and ubiquitous in research/education; it does not carry a formal OSI license. We download it at train time rather than vendoring it. ⚠️ The *parent* 80 Million Tiny Images dataset was withdrawn by its authors in 2020 over problematic content; CIFAR-10 itself remains widely used, but verify current terms at the source if you redistribute the data itself. |

**Bottom line:** ✅ Safe for open-source release. Code is original; weights are self-trained; dataset is downloaded at runtime, not redistributed.

> As with every entry in this repo: code license ≠ weights license ≠ dataset license. This note reflects the state at authoring and should be re-checked before publishing.
