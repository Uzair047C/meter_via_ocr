#!/usr/bin/env python3
"""
Confusion matrix (raw counts, NOT normalized) for the segmented single-digit CNN
(custom_q.keras) on the first 212 images of cnn-test/test.csv.

Folder layout expected (same as the repo's cnn-test/):
    cnn-test/
        custom_q.keras
        test.csv          # columns: image,label   (label = reading WITHOUT leading zeros)
        test/             # 5-digit ROI crops
        plot_cnn_confusion_matrix.py   <- this file

Run:   python plot_cnn_confusion_matrix.py
Needs: tensorflow (or tensorflow-cpu), opencv-python, numpy, pandas, matplotlib, seaborn, scikit-learn
"""
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "custom_q.keras"
CSV_PATH = HERE / "test.csv"
IMG_DIR = HERE / "test"
OUT_PNG = HERE / "cnn_confusion_matrix_212.png"

N_IMAGES = 212        # only the first 212 rows of test.csv
N_DIGITS = 5          # 5-digit meter reading
DIGIT_SIZE = 32       # model input is 32 x 32 x 1 (grayscale)
CLASSES = list(range(10))


# ---------------------------------------------------------------- helpers
def load_model():
    import keras
    return keras.models.load_model(str(MODEL_PATH), compile=False)


def split_into_digits(gray, n=N_DIGITS):
    """Fixed-width slicing of the ROI into n equal vertical strips."""
    h, w = gray.shape
    edges = np.linspace(0, w, n + 1).round().astype(int)
    return [gray[:, edges[i]:edges[i + 1]] for i in range(n)]


def prep_digit(d):
    d = cv2.resize(d, (DIGIT_SIZE, DIGIT_SIZE))
    return (d.astype(np.float32) / 255.0)[..., None]


# ---------------------------------------------------------------- main
def main():
    model = load_model()

    df = pd.read_csv(CSV_PATH, dtype={"label": str}).head(N_IMAGES)
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        gray = cv2.imread(str(IMG_DIR / row["image"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skipping unreadable image: {row['image']}")
            continue
        digits = np.stack([prep_digit(d) for d in split_into_digits(gray)])
        pred = model.predict(digits, verbose=0).argmax(axis=1)

        truth = str(row["label"]).strip().zfill(N_DIGITS)   # 3 -> "00003"
        y_true.extend(int(c) for c in truth)
        y_pred.extend(int(p) for p in pred)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)   # raw counts
    acc = accuracy_score(y_true, y_pred)
    print(f"Digits evaluated: {len(y_true)}  |  per-digit accuracy: {acc * 100:.2f}%")

    # ------------------------------------------------------------ plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=CLASSES, yticklabels=CLASSES,
        annot_kws={"size": 28, "weight": "bold"},     # big numbers inside cells
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_title(f"Confusion Matrix - first {len(df)} images "
                 f"({len(y_true)} digits), accuracy {acc * 100:.2f}%",
                 fontsize=18, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Digit", fontsize=16, fontweight="bold")
    ax.set_ylabel("True Digit", fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()