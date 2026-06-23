# License notes — LeNet-5

| Layer | Status |
|---|---|
| **Paper** | LeCun et al. (1998), *Gradient-Based Learning Applied to Document Recognition* — academic paper, freely available. |
| **Architecture** | LeNet-5 is a foundational architecture with no patent/license encumbrance for reimplementation. This implementation is original code written for paper2code2api. |
| **Weights** | Trained from scratch here on MNIST → produced by this repo, no upstream weight license to inherit. |
| **Dataset (MNIST)** | Released by Yann LeCun & Corinna Cortes. Widely used for research/education; commonly cited as available under a permissive/CC-style arrangement. Verify current terms at the source if redistributing the data itself (we download it at train time rather than vendoring it). |

**Bottom line:** ✅ Safe for open-source release. Code is original; weights are self-trained; dataset is downloaded at runtime, not redistributed.

> As with every entry in this repo: code license ≠ weights license ≠ dataset license. This note reflects the state at authoring and should be re-checked before publishing.
