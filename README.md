# paper2code2api

> **A hands-on course in computer vision — learn by rebuilding the papers that built the field.**
> For each landmark paper you'll understand the idea in plain English, build the model yourself, train it, and wrap it in a working API.

Most people learn deep learning backwards: they import a black box and never see what's inside. This course does the opposite. You start at the very first convolutional network (1998) and work forward, **reimplementing each paper from scratch** with heavily-commented code and a beginner-friendly lesson — then turning each model into a callable API so you can actually *use* what you built.

Every lesson does the thing that usually stops people cold: it **maps the paper's actual equations to the exact lines of code that implement them.** That paper-math-to-code leap is the real skill — and the whole point of this course.

Think of it as a textbook where every chapter ends with a running program.

---

## How to use this as a course

1. **Start at Stage 1, Lesson 1** and work down in order — each paper builds on ideas from the last.
2. **Open the lesson README** in each paper's folder. It teaches the concept from scratch — no prior deep-learning knowledge assumed.
3. **Build it yourself.** Each lesson walks through the code (`model.py`, `train.py`, `infer.py`, `api.py`) and ends with exercises.
4. **Run it.** Train the model, then start the API and send it your own images.

**Prerequisites:** basic Python (functions, classes, loops). Math is explained in plain language as it comes up — you do *not* need a math degree to start.

---

## 📚 Table of Contents — the curriculum

Lessons marked ✅ are written and runnable; the rest are planned and added over time.

### Stage 1 · Foundations

| # | Lesson | What you'll build | Status |
|---|---|---|---|
| 1 | [**LeNet-5** (1998) — Your First Convolutional Network](papers/stage1-foundations/lenet5) | A handwritten-digit classifier (MNIST) + a `POST /predict` API | ✅ |
| 2 | AlexNet (2012) — Deep Learning Goes Mainstream | Image classifier on real photos | 🔜 |
| 3 | VGG (2014) — Going Deeper with Small Filters | Classifier / feature extractor | 🔜 |

### Stage 2 · Core Architectures

| # | Lesson | What you'll build | Status |
|---|---|---|---|
| – | ResNet, U-Net, MobileNet, EfficientNet … | classifiers / backbones | 🔜 |

> Stages 3–6 (Detection & Segmentation, Generative, Transformers & Multimodal, Advanced) plus specialized tracks (OCR, Video, Pose, Face, Medical, Efficiency) are planned and will be added as lessons over time.

---

## What every lesson folder contains

| File | What it's for |
|---|---|
| `README.md` | **The lesson** — explains the paper for beginners and walks you through building it |
| `model.py` | The neural network, implemented from scratch and commented |
| `train.py` | Trains the model on its dataset |
| `infer.py` | Runs a prediction on a single image |
| `api.py` | A FastAPI server exposing the shared `POST /predict` contract |
| `make_examples.py` | Generates the input/output example images |
| `make_figures.py` | Generates the teaching diagrams (architecture, convolution, pooling, feature maps) |
| `<model>.pt` | Pretrained weights, shipped so `infer.py`/`api.py` work the moment you clone — retrain anytime with `train.py` |
| `LICENSE-NOTES.md` | License status for that paper's code, weights, and data |
| `requirements.txt` | What to `pip install` |

## The shared API contract

Every model speaks the same language, so they're interchangeable:

```
POST /predict   (multipart image)  -> JSON result
GET  /health                       -> { status, model_loaded }
```

---

## Quick taste — Lesson 1 output

The very first lesson trains LeNet-5 to read handwritten digits at ~99% accuracy:

![LeNet-5 inputs and predictions](papers/stage1-foundations/lenet5/assets/examples_grid.png)

→ **[Start Lesson 1: LeNet-5](papers/stage1-foundations/lenet5)**

---

## Licensing

Course code is **MIT** (see [`LICENSE`](LICENSE)). Third-party paper weights, datasets, and official implementations carry **their own** licenses — some non-commercial or copyleft. Always check each lesson's `LICENSE-NOTES.md` before redistributing or serving anything you didn't train yourself.

## Status

🚧 Early and active. LeNet-5 is the first full lesson and the template for the rest. Contributions welcome — pick a paper, follow the lesson template, open a PR.
