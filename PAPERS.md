# paper2code2api — Curriculum & Paper Index

> An open-source effort to turn landmark computer-vision papers into clean **reference implementations** and callable **inference APIs**. Every paper → readable code → a standard API endpoint.

This document is the master index: a beginner→advanced spine plus specialized tracks.

## Legend

- **Diff** — difficulty to reimplement from scratch: 🟢 easy · 🟡 medium · 🔴 hard
- **Data** — dataset to train/demo on
- **License** — code/weights license. ⚠️ = restriction that matters for open source (**verify per-repo before shipping** — code license ≠ weights license ≠ training-data license)
- **API** — the natural inference endpoint shape

> ⚠️ **All license/dataset notes below are best-effort recollections and MUST be verified per-repo before publishing.** They change over time, and the three license layers (code / weights / data) often differ.

---

## Stage 1 — Foundations

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| 1 | LeNet-5 — *Gradient-Based Learning Applied to Document Recognition* (1998) | 🟢 | MNIST | public | classify digit |
| 2 | AlexNet — *ImageNet Classification with Deep CNNs* (2012) | 🟢 | ImageNet | BSD (torchvision) | classify |
| 3 | VGG — *Very Deep Convolutional Networks* (2014) | 🟢 | ImageNet | BSD | classify / features |
| 4 | Network in Network (2013) | 🟢 | CIFAR | public | concept (1×1 conv, GAP) |
| 5 | Batch Normalization (2015) | 🟢 | ImageNet | public | technique |
| 6 | Dropout (2014) | 🟢 | — | public | technique |
| 7 | Adam — *A Method for Stochastic Optimization* (2014) | 🟢 | — | public | technique |
| 8 | He initialization — *Delving Deep into Rectifiers* (2015) | 🟢 | ImageNet | public | technique |

## Stage 2 — Core architectures

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| 9 | GoogLeNet / Inception — *Going Deeper with Convolutions* (2014) | 🟡 | ImageNet | Apache/BSD | classify |
| 10 | **ResNet** ⭐ — *Deep Residual Learning* (2015) | 🟢 | ImageNet | BSD | classify / backbone |
| 11 | U-Net — *Convolutional Networks for Biomedical Image Segmentation* (2015) | 🟢 | ISBI cells | public | segment |
| 12 | DenseNet — *Densely Connected Convolutional Networks* (2017) | 🟡 | ImageNet | BSD | classify |
| 13 | ResNeXt — *Aggregated Residual Transformations* (2016) | 🟡 | ImageNet | BSD | classify |
| 14 | MobileNet v1/v2 (2017–18) | 🟢 | ImageNet | Apache | edge classify |
| 15 | EfficientNet (2019) | 🟡 | ImageNet | Apache | classify |
| 16 | SENet — *Squeeze-and-Excitation Networks* (2017) | 🟢 | ImageNet | Apache | classify (attention block) |
| 17 | Xception (2016) | 🟡 | ImageNet | MIT | classify |
| 18 | ConvNeXt (2022) | 🟡 | ImageNet | MIT | modern CNN baseline |

## Stage 3 — Detection & segmentation

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| 19 | R-CNN → Fast → Faster R-CNN (2014–15) | 🔴 | PASCAL VOC | BSD | detect boxes |
| 20 | YOLO v1 — *You Only Look Once* (2015) | 🟡 | VOC/COCO | public (original) | real-time detect |
| 21 | SSD — *Single Shot MultiBox Detector* (2016) | 🟡 | COCO | Apache | detect |
| 22 | Feature Pyramid Networks (2016) | 🟡 | COCO | Apache | multi-scale features |
| 23 | RetinaNet — *Focal Loss for Dense Object Detection* (2017) | 🟡 | COCO | Apache | dense detect |
| 24 | Mask R-CNN (2017) | 🔴 | COCO | ⚠️ varies (Detectron2 = Apache) | instance masks |
| 25 | FCN — *Fully Convolutional Networks* (2015) | 🟢 | VOC | public | semantic segment |
| 26 | DeepLab v3+ (2018) | 🟡 | Cityscapes | Apache | scene parse |
| 27 | CenterNet — *Objects as Points* (2019) | 🟡 | COCO | MIT | anchor-free detect |

## Stage 4 — Generative

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| 28 | GAN — *Generative Adversarial Networks* (2014) | 🟢 | MNIST | public | generate |
| 29 | DCGAN (2015) | 🟢 | LSUN/faces | MIT | generate |
| 30 | pix2pix — *Image-to-Image Translation with cGANs* (2016) | 🟡 | maps/facades | ⚠️ BSD, non-commercial weights | img→img |
| 31 | CycleGAN (2017) | 🟡 | horse↔zebra | ⚠️ as pix2pix | unpaired translate |
| 32 | StyleGAN2 (2019) | 🔴 | FFHQ | ⚠️ Nvidia source-available | face generate |
| 33 | VAE — *Auto-Encoding Variational Bayes* (2013) | 🟢 | MNIST | public | encode/generate |
| 34 | DDPM — *Denoising Diffusion Probabilistic Models* (2020) | 🔴 | CIFAR/faces | Apache-ish | diffusion generate |
| 35 | Latent Diffusion / Stable Diffusion (2021) | 🔴 | LAION | ⚠️ CreativeML OpenRAIL | text→image |

## Stage 5 — Transformers & multimodal

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| 36 | Attention Is All You Need (2017) | 🟡 | — (NLP) | public | background |
| 37 | Vision Transformer (ViT) (2020) | 🟡 | ImageNet/JFT | Apache | classify |
| 38 | DETR — *End-to-End Object Detection with Transformers* (2020) | 🔴 | COCO | Apache | detect (no NMS) |
| 39 | Swin Transformer (2021) | 🔴 | ImageNet/COCO | MIT | backbone / detect |
| 40 | MAE — *Masked Autoencoders* (2021) | 🟡 | ImageNet | ⚠️ CC-BY-NC (Meta) | self-supervised pretrain |
| 41 | CLIP (2021) | 🟡 | WIT (web) | MIT | zero-shot classify / search |
| 42 | Segment Anything (SAM) (2023) | 🟡 | SA-1B | Apache 2.0 | prompt→mask |
| 43 | DINOv2 (2023) | 🟡 | curated web | Apache 2.0 | universal features |

## Stage 6 — Advanced

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| 44 | NeRF — *Neural Radiance Fields* (2020) | 🔴 | synthetic/LLFF | MIT | novel-view render |
| 45 | 3D Gaussian Splatting (2023) | 🔴 | multi-view | ⚠️ research-only | real-time 3D render |
| 46 | YOLOv8 / Ultralytics (2023) | 🟡 | COCO | ⚠️ **AGPL-3.0** (copyleft) | detect/segment/pose |
| 47 | Grounding DINO (2023) | 🔴 | mixed | Apache | text→detect |
| 48 | SAM 2 (2024) | 🔴 | video | Apache | video segment/track |
| 49 | Depth Anything v2 (2024) | 🟡 | mixed | ⚠️ code Apache, large model CC-BY-NC | monocular depth |
| 50 | Florence-2 (2024) | 🟡 | FLD-5B | MIT | unified vision tasks |

---

# Specialized Tracks

## Track A — OCR & Document Understanding

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| A1 | CRNN — *An End-to-End Trainable Neural Network for Image-based Sequence Recognition* (2015) | 🟡 | Synth90k | MIT | text recognition |
| A2 | CTPN — *Detecting Text in Natural Image with Connectionist Text Proposal Network* (2016) | 🟡 | ICDAR | public | text detection |
| A3 | EAST — *An Efficient and Accurate Scene Text Detector* (2017) | 🟡 | ICDAR | public | text detection |
| A4 | DBNet — *Real-time Scene Text Detection with Differentiable Binarization* (2019) | 🟡 | ICDAR/Total-Text | Apache (PaddleOCR) | text detection |
| A5 | LayoutLM v1/v2/v3 (2019–22) | 🟡 | IIT-CDIP/FUNSD | ⚠️ v3 CC-BY-NC | doc understanding |
| A6 | TrOCR — *Transformer-based OCR with Pre-trained Models* (2021) | 🟡 | SROIE/IAM | MIT | OCR (printed/handwritten) |
| A7 | Donut — *OCR-free Document Understanding Transformer* (2021) | 🟡 | DocVQA/CORD | MIT | doc parse → JSON |

## Track B — Video Understanding

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| B1 | Two-Stream Convolutional Networks (2014) | 🟡 | UCF101 | public | action recognition |
| B2 | C3D — *Learning Spatiotemporal Features with 3D ConvNets* (2015) | 🟡 | Sports-1M/UCF101 | public | action recognition |
| B3 | TSN — *Temporal Segment Networks* (2016) | 🟡 | UCF101/Kinetics | BSD | action recognition |
| B4 | I3D — *Quo Vadis, Action Recognition?* (2017) | 🔴 | Kinetics | Apache | action recognition |
| B5 | Non-local Neural Networks (2017) | 🔴 | Kinetics | ⚠️ as Detectron | video features |
| B6 | SlowFast Networks (2018) | 🔴 | Kinetics/AVA | Apache (PySlowFast) | action recognition/detection |
| B7 | TimeSformer — *Is Space-Time Attention All You Need for Video?* (2021) | 🔴 | Kinetics | ⚠️ CC-BY-NC (Meta) | video classify |
| B8 | VideoMAE (2022) | 🔴 | Kinetics/SSv2 | CC-BY-NC / Apache code | self-supervised video |

## Track C — Pose & Keypoint Estimation

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| C1 | DeepPose (2014) | 🟡 | LSP/FLIC | public | human pose |
| C2 | Convolutional Pose Machines (2016) | 🟡 | MPII | ⚠️ non-commercial | human pose |
| C3 | Stacked Hourglass Networks (2016) | 🟡 | MPII/COCO | BSD/MIT | human pose |
| C4 | OpenPose — *Realtime Multi-Person 2D Pose* (2017) | 🔴 | COCO | ⚠️ **non-commercial** (academic only) | multi-person pose |
| C5 | HRNet — *Deep High-Resolution Representation Learning* (2019) | 🔴 | COCO/MPII | MIT | pose / segment / detect |
| C6 | BlazePose / MediaPipe Pose (2020) | 🟡 | custom | Apache | real-time on-device pose |
| C7 | ViTPose (2022) | 🔴 | COCO | Apache | pose (transformer) |

## Track D — Face

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| D1 | DeepFace (2014) | 🟡 | SFC | research | face verification |
| D2 | FaceNet — *A Unified Embedding for Face Recognition* (2015) | 🟡 | LFW/youtube | MIT (reimpls) | face embedding/match |
| D3 | MTCNN — *Joint Face Detection and Alignment* (2016) | 🟢 | WIDER FACE | MIT | face detect + align |
| D4 | SphereFace (2017) | 🟡 | CASIA-WebFace | MIT | face recognition |
| D5 | CosFace (2018) | 🟡 | CASIA/MS-Celeb | research | face recognition |
| D6 | ArcFace — *Additive Angular Margin Loss* (2018) | 🟡 | MS-Celeb-1M | MIT | face embedding/match |
| D7 | RetinaFace (2019) | 🟡 | WIDER FACE | MIT | face detect + landmarks |

> ⚠️ Face recognition has **ethical/legal/privacy** considerations (biometrics, consent, dataset provenance — several common face datasets have been retracted). Add a clear use-policy + dataset-sourcing note in the repo for this track.

## Track E — Medical Imaging

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| E1 | U-Net (2015) | 🟢 | ISBI | public | 2D segmentation |
| E2 | V-Net — *Fully Convolutional Networks for Volumetric Segmentation* (2016) | 🟡 | PROMISE12 | public | 3D segmentation |
| E3 | 3D U-Net (2016) | 🟡 | Xenopus kidney | public | volumetric segment |
| E4 | Attention U-Net (2018) | 🟡 | CT (pancreas) | MIT | segmentation |
| E5 | CheXNet — *Radiologist-Level Pneumonia Detection* (2017) | 🟢 | ChestX-ray14 | research | classify (X-ray) |
| E6 | nnU-Net — *Self-configuring Method for Biomedical Segmentation* (2020) | 🔴 | Medical Decathlon | Apache | auto-config segment |
| E7 | TransUNet (2021) | 🔴 | Synapse multi-organ | Apache | segmentation (transformer) |
| E8 | MedSAM — *Segment Anything in Medical Images* (2023) | 🟡 | multi-modality | Apache | prompt→mask (medical) |

> ⚠️ Medical models are **not clinical devices**. Repo must state "research/education only, not for diagnosis." Mind dataset licenses (PHI/HIPAA, data-use agreements).

## Track F — Efficiency & Deployment

| # | Paper (year) | Diff | Data | License | API |
|---|---|---|---|---|---|
| F1 | Knowledge Distillation — *Distilling the Knowledge in a Neural Network* (Hinton, 2015) | 🟢 | any | public | technique |
| F2 | Learning both Weights and Connections (Han, 2015) | 🟢 | ImageNet | public | pruning technique |
| F3 | Deep Compression (Han, 2015) | 🟡 | ImageNet | public | prune+quantize+huffman |
| F4 | XNOR-Net — *Binary Convolutional Networks* (2016) | 🔴 | ImageNet | BSD | binary inference |
| F5 | ShuffleNet (2017) | 🟡 | ImageNet | Apache | efficient classify |
| F6 | Quantization for Integer-Arithmetic-Only Inference (Jacob, 2017) | 🟡 | ImageNet | Apache | INT8 inference |
| F7 | The Lottery Ticket Hypothesis (Frankle & Carbin, 2018) | 🟡 | MNIST/CIFAR | MIT | pruning research |
| F8 | Once-for-All — *Train One Network, Specialize for Deployment* (2019) | 🔴 | ImageNet | MIT | NAS / deploy |

> Deployment tooling (not papers, but the "2api" backbone): ONNX, TensorRT, OpenVINO, TorchScript, Triton Inference Server, FastAPI/BentoML for serving.

---

## Licensing landmines (read before publishing)

The single biggest risk for an open-source paper2code2api repo:

1. **Ultralytics YOLOv5/v8 = AGPL-3.0.** AGPL's network-use clause can force you to open-source your **entire serving stack** if your API serves it. For a clean permissive detection reference, prefer original **YOLO**, **DETR**, **RetinaNet**, or **CenterNet**.
2. **Non-commercial weights/code** (⚠️ above): OpenPose, AlphaPose, MAE, TimeSformer, VideoMAE weights, LayoutLMv3, StyleGAN2, 3D Gaussian Splatting. Fine to *reference/study*, risky to *redistribute or serve*.
3. **OpenRAIL (Stable Diffusion)** carries use restrictions — not OSI "open source."
4. **Dataset licenses ≠ model licenses.** ImageNet, LAION, and several face datasets have their own (sometimes retracted) terms.
5. **Code license ≠ weights license.** Many repos are Apache code + non-commercial weights.

**Action:** run a verification pass on every ⚠️ row against the *current* upstream license before including it.

---

## Suggested repo conventions

- One folder per paper: `papers/<stage>/<slug>/` containing `README.md` (paper summary + math), `model.py` (clean impl), `train.py`, `infer.py`, `api.py` (standard endpoint), `LICENSE-NOTES.md`.
- A **single shared API contract** so every paper exposes the same shape (e.g. `POST /predict` with image bytes → JSON). Pick REST (FastAPI) as the baseline.
- Each paper card: difficulty, dataset, license status, "good first issue?" flag.
