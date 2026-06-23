# LeNet-5 — Handwritten Digit Classification

> **Stage 1 · Foundations** · Difficulty 🟢 · Dataset: MNIST · License: ✅ public (see `LICENSE-NOTES.md`)

The paper that started it all. LeCun et al. (1998) showed that a convolutional
neural network trained end-to-end with backpropagation could read handwritten
digits — convolution + pooling + fully-connected, the template every CNN since
has followed.

**Paper:** *Gradient-Based Learning Applied to Document Recognition*, LeCun, Bottou, Bengio, Haffner, Proc. IEEE 1998.

## The idea in one minute

- **Local receptive fields + weight sharing (convolution):** detect a feature
  anywhere in the image with far fewer parameters than a dense layer.
- **Subsampling (pooling):** build tolerance to small shifts/distortions.
- **Hierarchy:** stack conv→pool blocks so later layers see larger, more abstract patterns.
- Train the whole thing by gradient descent — no hand-engineered features.

## Architecture

```
input 1x32x32
 └ C1  conv 5x5, 6 maps   -> 6x28x28   + tanh
 └ S2  avgpool 2x2        -> 6x14x14
 └ C3  conv 5x5, 16 maps  -> 16x10x10  + tanh
 └ S4  avgpool 2x2        -> 16x5x5
 └ C5  conv 5x5, 120 maps -> 120x1x1   + tanh
 └ F6  fc 120 -> 84       + tanh
 └ out fc 84 -> 10
```

~61.7k parameters.

### Faithful vs. modernized

This is the common modern reproduction. Two deliberate simplifications vs. the 1998 paper:

1. **C3 connections.** The original used a hand-designed sparse table connecting only
   some S2 maps to each C3 map. We use full connections (simpler, negligible cost at this scale).
2. **Output layer.** The original used Gaussian/RBF units with a specialized loss.
   We use a standard linear layer + softmax cross-entropy.

We *do* keep the period-authentic choices of **tanh** activations, **average** pooling,
and the **32×32** input (MNIST 28×28 padded by 2).

## Files

| File | Purpose |
|---|---|
| `model.py` | LeNet-5 architecture (the reference implementation) |
| `train.py` | Train on MNIST, save `lenet5.pt` |
| `infer.py` | Preprocess + predict; usable standalone or as a library |
| `api.py` | FastAPI server exposing the shared `POST /predict` contract |
| `requirements.txt` | Dependencies |
| `LICENSE-NOTES.md` | License status (✅ safe to ship) |

## Quickstart

```bash
pip install -r requirements.txt

# 1) train (downloads MNIST on first run; ~2 min on CPU, expect ~99% test acc)
python train.py --epochs 5

# 2) sanity-check a single image from the CLI
python infer.py some_digit.png            # MNIST-style white-on-black
python infer.py photo_of_digit.png --invert   # black-on-white drawing/photo

# 3) serve the API
uvicorn api:app --reload
# open http://127.0.0.1:8000/docs
```

## API contract

`POST /predict` — multipart form, field `file` = image; optional `?invert=true`.

```jsonc
// response
{
  "prediction": 7,
  "confidence": 0.9991,
  "probabilities": [/* 10 floats, index = digit */]
}
```

`GET /health` → `{ "status": "ok", "model_loaded": true }`

Example:

```bash
curl -F "file=@some_digit.png" "http://127.0.0.1:8000/predict?invert=false"
```

## Note on input orientation

MNIST is **white digit on black background**, normalized. Real photos/drawings are
usually the opposite, so pass `invert=true` (CLI `--invert`) for those. The classifier
is only as good as the preprocessing match — this is the #1 cause of bad predictions on
custom images.
