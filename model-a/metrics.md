# Model A - Results (CS663 Project 2, Dog Training Assistant)

**Model:** MobileNetV3Small (ImageNet weights), student fine-tuned - as
specified in the proposal (Model A = MobileNetV3, lightweight CNN).
Input 224x224, batch 32, seed 42, CPU training (no GPU). The Keras
backbone already contains its input Rescaling layer, so the exported
model takes raw 0-255 pixels.

**Data:** dataset v4, split exactly as recorded in
`real-data-v4/provenance.json` (source-level split, no reshuffling):
train 626 / validation 139 / test 235, 250 per class overall.

## Training

| Phase | What | LR | Epochs run | Notes |
|---|---|---|---|---|
| 1 | backbone frozen, new head (GAP, dropout 0.3, dense 4) | 0.001 | 15/20 | augmentation: flip, rotation, zoom, brightness; early stopping on val accuracy |
| 2 | top 30 backbone layers unfrozen (BN frozen) | 0.0001 | 8/8 | fine-tune |

Curves: `curves.png` (accuracy + loss, train vs validation, all 23 epochs).
Full per-epoch numbers: `history.json`.

## Held-out test results (235 images, evaluated once)

- **Test accuracy: 0.7489 (176/235)**
- Validation accuracy: 0.7770

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| sitting | 0.643 | 0.871 | 0.740 | 62 |
| standing | 0.721 | 0.633 | 0.674 | 49 |
| lying_down | 0.867 | 0.839 | 0.853 | 62 |
| walking | 0.812 | 0.629 | 0.709 | 62 |

Confusion matrix (rows = true, cols = predicted; order sitting, standing, lying_down, walking):

```
[[54, 2, 4, 2], [11, 31, 2, 5], [7, 1, 52, 2], [12, 9, 2, 39]]
```
Heatmap: `confusion.png`. 10 individual test cases: `sample_predictions.md`.

Reading the matrix: lying_down is the strongest class (52/62, F1 0.85).
The confusions are the honest weak spots - standing is mistaken for
sitting (11) and walking (5); walking is mistaken for sitting (12) and
standing (9). Single frames of standing/walking/sitting genuinely
overlap; this is one reason the full project judges commands from a
*sequence* of frames (video-sequence-layer), not one frame.

## Exports and verification

- `model_a.keras` - inference model (backbone + head). Verified to
  reload standalone with a plain `keras.models.load_model` call.
- `model_a.tflite` - float16-quantised, 1.95 MB, for the Android app.
  Verified with the TF Lite interpreter on a held-out test image
  (`tflite_check.json`): input [1, 224, 224, 3] float32, output
  probabilities sum to 1.0, prediction lying_down @ 0.999 agrees with
  the Keras model (file `ph_basile_morin_lying_down_001.jpg`, true
  label lying_down).
- `train_model_a.py` is the exact script that produced everything above.
