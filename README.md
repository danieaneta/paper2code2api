# paper2code2api

> Turning landmark **computer-vision papers** into clean **reference implementations** and callable **inference APIs**.
> Every paper → readable code → a standard API endpoint. Open source, learn-by-building.

Most "paper implementations" are either a research dump you can't run or a black-box library you can't learn from. **paper2code2api** aims for both: each paper gets a from-scratch, well-commented implementation *and* a uniform REST API so you can actually call it.

## How it's organized

- **[`PAPERS.md`](PAPERS.md)** — the master curriculum: ~50 papers across 6 stages (Foundations → Advanced) plus specialized tracks (OCR, Video, Pose, Face, Medical, Efficiency). Each entry is annotated with difficulty, dataset, license status, and the natural API shape.
- **`papers/<stage>/<slug>/`** — one self-contained folder per paper.

Every paper folder follows the same layout:

| File | Purpose |
|---|---|
| `README.md` | Paper summary + the idea + architecture |
| `model.py` | The reference implementation |
| `train.py` | Reproduce / train weights |
| `infer.py` | Preprocess + predict (CLI + library) |
| `api.py` | FastAPI server — the shared `POST /predict` contract |
| `LICENSE-NOTES.md` | Per-paper license status |
| `requirements.txt` | Dependencies |

## The shared API contract

Every paper exposes the same baseline so they're interchangeable:

```
POST /predict   (multipart image)  -> JSON result
GET  /health                       -> { status, model_loaded }
```

## Implemented so far

| Stage | Paper | Status | Demo |
|---|---|---|---|
| 1 · Foundations | [**LeNet-5** (1998)](papers/stage1-foundations/lenet5) | ✅ trained (98.7% MNIST) + API | `POST /predict` → digit |

### Try LeNet-5 in 3 commands

```bash
cd papers/stage1-foundations/lenet5
pip install -r requirements.txt
uvicorn api:app --reload     # weights (lenet5.pt) are included — works out of the box
# open http://127.0.0.1:8000/docs
```

```bash
curl -F "file=@your_digit.png" "http://127.0.0.1:8000/predict?invert=false"
# -> {"prediction": 7, "confidence": 0.9998, "probabilities": [...]}
```

## Licensing

Original code in this repo is **MIT** (see [`LICENSE`](LICENSE)). But **third-party paper weights, datasets, and official implementations carry their own licenses** — some are non-commercial or copyleft (e.g. AGPL). Always check the per-paper `LICENSE-NOTES.md` and the licensing section of [`PAPERS.md`](PAPERS.md) before redistributing or serving anything you didn't train yourself here.

## Roadmap

Working down the curriculum in [`PAPERS.md`](PAPERS.md), Stage 1 first. Next up: AlexNet, VGG, ResNet, U-Net. Contributions welcome — pick an unimplemented paper, follow the folder template, open a PR.

## Status

🚧 Early and active. LeNet-5 is the reference template; the rest of the curriculum is mapped and ready to build.
