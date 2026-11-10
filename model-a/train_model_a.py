#!/usr/bin/env python3
"""Train Model A for CS663 Project 2 - Dog Training Assistant.

Model A per the group proposal (section 5): a lightweight CNN
(MobileNetV3) fine-tuned by the students for frame-level dog posture
classification - sitting / standing / lying_down / walking.

Implementation: Keras MobileNetV3Small (ImageNet weights), input
224x224, batch 32, CPU training.
  Phase 1: frozen backbone + new head (GAP -> Dropout -> Dense 4
           softmax), Adam 1e-3, up to 20 epochs, augmentation
           (flip / rotation / zoom / brightness), early stopping on
           val_accuracy (patience 4, restore best).
  Phase 2: unfreeze the top backbone layers (BatchNorm kept frozen),
           Adam 1e-4, up to 8 epochs, early stopping (patience 3).

Data: dataset v4 (1,000 images, 250/class). File lists are built ONLY
from real-data-v4/provenance.json - the recorded source-level split
(train 626 / validation 139 / test 235) is used exactly as published;
nothing is reshuffled, so no source (uploader/video) leaks across
splits. The test set is evaluated exactly once, at the very end.

Run:  ~/tfvenv/bin/python train_model_a.py
Outputs (this folder): history.json, curves.png, confusion.png,
metrics.json, metrics.md, sample_predictions.md, model_a.keras,
model_a.tflite.
"""
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras
from keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "gemini-finetune"
PROVENANCE = DATA_ROOT / "real-data-v4" / "provenance.json"

IMG_SIZE = 224
BATCH = 32
CLASS_NAMES = ["sitting", "standing", "lying_down", "walking"]  # project order
CLASS_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 8
PHASE2_UNFREEZE = 30          # top backbone layers (non-BN counted below)
DROPOUT = 0.3


# ---------------------------------------------------------------- data
def load_split(split):
    prov = json.load(open(PROVENANCE))
    rows = [r for r in prov if r["split"] == split]
    paths = [str(DATA_ROOT / r["dest"]) for r in rows]
    labels = [CLASS_IDX[r["behaviour"]] for r in rows]
    for p in paths:
        assert os.path.exists(p), p
    return paths, np.array(labels, dtype=np.int32), rows


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, label  # float 0..255; preprocessing lives in the model

    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- model
def build_models():
    backbone = keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
        weights="imagenet")
    backbone.trainable = False

    # Inference/export core: backbone -> head. NOTE on preprocessing:
    # Keras MobileNetV3Small already contains a Rescaling(1/127.5, -1)
    # layer as its first layer and mobilenet_v3.preprocess_input is an
    # identity passthrough (both verified) - so the core takes raw
    # 0..255 pixels and must NOT rescale again (double rescaling crushes
    # the input to a near-constant and training collapses to chance).
    core_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_0_255")
    x = core_in
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(DROPOUT, name="dropout")(x)
    out = layers.Dense(len(CLASS_NAMES), activation="softmax",
                       name="posture")(x)
    core = keras.Model(core_in, out, name="model_a_core")

    # Training model adds augmentation in front of the core.
    train_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="train_image")
    aug = keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.08, seed=SEED),
        layers.RandomZoom(0.10, seed=SEED),
        layers.RandomBrightness(0.15, seed=SEED),
    ], name="augmentation")
    train_model = keras.Model(train_in, core(aug(train_in)),
                              name="model_a_train")
    return backbone, core, train_model


def compile_and_fit(model, train_ds, val_ds, epochs, lr, tag, ckpt):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    cbs = [keras.callbacks.EarlyStopping(
               monitor="val_accuracy", patience=4 if tag == "phase1" else 3,
               restore_best_weights=True, verbose=1),
           keras.callbacks.ModelCheckpoint(
               ckpt, monitor="val_accuracy", save_best_only=True, verbose=0)]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=cbs, verbose=2)
    print(f"[{tag}] fit took {time.time() - t0:.0f}s, "
          f"epochs run: {len(hist.history['loss'])}")
    return hist.history


def main():
    splits = {s: load_split(s) for s in ("train", "validation", "test")}
    for s, (p, l, _) in splits.items():
        counts = {CLASS_NAMES[c]: int((l == c).sum()) for c in range(4)}
        print(f"{s}: {len(p)} images {counts}")
    train_ds = make_ds(*splits["train"][:2], training=True)
    val_ds = make_ds(*splits["validation"][:2], training=False)
    test_ds = make_ds(*splits["test"][:2], training=False)

    backbone, core, model = build_models()
    print(f"backbone: MobileNetV3Small, layers={len(backbone.layers)}, "
          f"trainable params (phase 1): "
          f"{sum(int(np.prod(v.shape)) for v in model.trainable_variables)}")

    history = {"phase1": None, "phase2": None}
    ckpt = str(HERE / "tmp" / "best.keras")

    history["phase1"] = compile_and_fit(
        model, train_ds, val_ds, PHASE1_EPOCHS, 1e-3, "phase1", ckpt)

    # ---- phase 2: fine-tune top of the backbone ----------------------
    backbone.trainable = True
    unfrozen = 0
    for layer in backbone.layers:
        layer.trainable = False
    for layer in reversed(backbone.layers):
        if isinstance(layer, layers.BatchNormalization):
            continue                      # BN stays frozen (small data)
        layer.trainable = True
        unfrozen += 1
        if unfrozen >= PHASE2_UNFREEZE:
            break
    print(f"phase 2: unfroze {unfrozen} top backbone layers (BN frozen), "
          f"trainable params: "
          f"{sum(int(np.prod(v.shape)) for v in model.trainable_variables)}")
    history["phase2"] = compile_and_fit(
        model, train_ds, val_ds, PHASE2_EPOCHS, 1e-4, "phase2", ckpt)

    merged = {
        "epochs_phase1": len(history["phase1"]["loss"]),
        "epochs_phase2": len(history["phase2"]["loss"]),
        "loss": history["phase1"]["loss"] + history["phase2"]["loss"],
        "accuracy": (history["phase1"]["accuracy"]
                     + history["phase2"]["accuracy"]),
        "val_loss": (history["phase1"]["val_loss"]
                     + history["phase2"]["val_loss"]),
        "val_accuracy": (history["phase1"]["val_accuracy"]
                         + history["phase2"]["val_accuracy"]),
        "phases": history,
    }
    json.dump(merged, open(HERE / "history.json", "w"), indent=1, default=float)
    print(f"total epochs: {merged['epochs_phase1']}+{merged['epochs_phase2']}")

    # ---- evaluate the held-out test set exactly once ------------------
    test_paths, test_labels, test_rows = splits["test"]
    probs = core.predict(test_ds, verbose=0)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    test_acc = float((pred == test_labels).mean())
    val_probs = core.predict(val_ds, verbose=0)
    val_acc = float((val_probs.argmax(axis=1)
                     == splits["validation"][1]).mean())
    print(f"FINAL  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # confusion + per-class metrics (manual, no sklearn dependency here)
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(test_labels, pred):
        cm[t, p] += 1
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(cm[i, i])
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[name] = {"precision": round(float(prec), 4),
                           "recall": round(float(rec), 4),
                           "f1": round(float(f1), 4),
                           "support": int(cm[i, :].sum())}
    metrics = {
        "model": "Model A - MobileNetV3Small (ImageNet), student fine-tuned",
        "input_size": IMG_SIZE, "batch_size": BATCH, "seed": SEED,
        "dropout": DROPOUT,
        "phase1": {"epochs_run": merged["epochs_phase1"],
                   "max_epochs": PHASE1_EPOCHS, "lr": 1e-3,
                   "backbone": "frozen"},
        "phase2": {"epochs_run": merged["epochs_phase2"],
                   "max_epochs": PHASE2_EPOCHS, "lr": 1e-4,
                   "unfrozen_top_layers": int(unfrozen)},
        "counts": {"train": len(splits["train"][0]),
                   "validation": len(splits["validation"][0]),
                   "test": len(test_paths)},
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": CLASS_NAMES, "rows_true_cols_pred":
                             cm.tolist()},
        "class_order": CLASS_NAMES,
    }
    json.dump(metrics, open(HERE / "metrics.json", "w"), indent=1)

    np.savez(HERE / "tmp" / "test_eval.npz", probs=probs, pred=pred,
             conf=conf, labels=test_labels,
             paths=np.array(test_paths))
    json.dump([{"file": Path(r["dest"]).name, "dest": r["dest"],
                "source": r["source"], "true": CLASS_NAMES[t],
                "pred": CLASS_NAMES[p], "confidence": round(float(c), 4)}
               for r, t, p, c in zip(test_rows, test_labels, pred, conf)],
              open(HERE / "tmp" / "test_predictions.json", "w"), indent=1)

    # ---- exports -------------------------------------------------------
    core.save(HERE / "model_a.keras")
    print("saved model_a.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(core)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_bytes = converter.convert()
    (HERE / "model_a.tflite").write_bytes(tflite_bytes)
    print(f"saved model_a.tflite ({len(tflite_bytes)/1e6:.2f} MB, float16)")

    # verify the tflite artifact actually runs
    from PIL import Image
    interp = tf.lite.Interpreter(model_path=str(HERE / "model_a.tflite"))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]
    sample = np.asarray(
        Image.open(test_paths[0]).convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
        dtype=np.float32)[None]
    interp.set_tensor(inp["index"], sample)
    interp.invoke()
    tfl_out = interp.get_tensor(outp["index"])[0]
    check = {"tflite_input": str(inp["shape"]) + " " + str(inp["dtype"]),
             "sample_file": Path(test_paths[0]).name,
             "sample_true": CLASS_NAMES[int(test_labels[0])],
             "keras_pred": CLASS_NAMES[int(pred[0])],
             "tflite_pred": CLASS_NAMES[int(tfl_out.argmax())],
             "tflite_probs_sum": round(float(tfl_out.sum()), 4),
             "tflite_probs": [round(float(x), 4) for x in tfl_out],
             "agree_with_keras": bool(tfl_out.argmax() == pred[0])}
    json.dump(check, open(HERE / "tmp" / "tflite_check.json", "w"), indent=1)
    print("tflite check:", check)


if __name__ == "__main__":
    main()
