#!/usr/bin/env python3
"""
PaddleOCR test-set classification report (character-level).

Prints the per-class precision / recall / F1 / support table in the terminal and
saves a figure with per-class F1 bars, macro-averaged bars, overall-accuracy bars
and the confusion matrix -- the same layout as the paper's Table 5.

Usage:
  python paddle_classification_report.py                       # run OCR over the test set
  python paddle_classification_report.py --limit 8             # smoke test
  python paddle_classification_report.py --from-csv out.csv    # skip OCR, reuse predictions

Predictions are always cached to output/paddle_test_predictions.csv so re-runs are free.
"""

import argparse
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "tools"))
sys.path.insert(0, current_dir)

DIGITS = "0123456789"


def load_labels(label_file):
    """image filename -> ground-truth label."""
    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                labels[os.path.basename(parts[0].strip())] = parts[1].strip()
    return labels


def read_csv_predictions(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [(r["image"], r["gt"].strip(), (r["pred"] or "").strip(),
             float(r["confidence"] or 0)) for r in rows]


def run_ocr(image_dir, label_file, model_dir, char_dict, limit=None, batch_size=16):
    """Run the vendored PaddleOCR recognizer over the labeled test images."""
    from predict_rec import TextRecognizer
    from ppocr.utils.utility import check_and_read
    import utility1 as utility

    labels = load_labels(label_file)
    valid = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    files = sorted(
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid and f in labels
    )
    if limit:
        files = files[:limit]
    print(f"Running PaddleOCR on {len(files)} labeled test images...")

    args = utility.init_args().parse_args([
        "--use_gpu=False",
        f"--image_dir={image_dir}",
        f"--rec_model_dir={model_dir}",
        f"--rec_char_dict_path={char_dict}",
        "--rec_image_shape=3,48,320",
    ])
    recognizer = TextRecognizer(args)

    rows = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        imgs = []
        for fpath in batch_files:
            img, flag, _ = check_and_read(fpath)
            if not flag:
                img = cv2.imread(fpath)
            if img is not None:
                imgs.append((fpath, img))
        if not imgs:
            continue
        batch_results, _ = recognizer([im for _, im in imgs])
        for (fpath, _), (text, conf) in zip(imgs, batch_results):
            name = os.path.basename(fpath)
            rows.append((name, labels[name], text, float(conf)))
        print(f"  {min(i + batch_size, len(files))}/{len(files)}", end="\r")
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default=os.path.join(current_dir, "dataset", "images", "test"))
    ap.add_argument("--label-file", default=os.path.join(current_dir, "dataset", "labels", "test_list.txt"))
    ap.add_argument("--model-dir", default=os.path.join(current_dir, "models", "digit_only_rec"))
    ap.add_argument("--char-dict", default=os.path.join(current_dir, "digit_dict.txt"))
    ap.add_argument("--out", default=os.path.join(current_dir, "output"))
    ap.add_argument("--from-csv", help="reuse cached predictions instead of running OCR")
    ap.add_argument("--limit", type=int, help="only evaluate the first N images (smoke test)")
    ap.add_argument("--no-open", action="store_true", help="do not open the saved figure")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    csv_path = a.from_csv or os.path.join(a.out, "paddle_test_predictions.csv")
    if a.from_csv:
        rows = read_csv_predictions(a.from_csv)
        print(f"Loaded {len(rows)} cached predictions from {a.from_csv}")
    else:
        rows = run_ocr(a.image_dir, a.label_file, a.model_dir, a.char_dict, a.limit)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "gt", "pred", "confidence"])
            w.writerows(rows)
        print(f"Predictions cached to {csv_path}")

    gt = [r[1] for r in rows]
    pred = [r[2] for r in rows]

    # ---- image level ----
    n = len(rows)
    exact = sum(g == p for g, p in zip(gt, pred))
    same_len = [(g, p) for g, p in zip(gt, pred) if len(g) == len(p)]

    # ---- character level on the same-length subset (wrong length = dropped, as in the paper) ----
    y_true, y_pred = [], []
    for g, p in same_len:
        for gc, pc in zip(g, p):
            if gc in DIGITS and pc in DIGITS:
                y_true.append(gc)
                y_pred.append(pc)

    support_total = len(y_true)
    classes = [d for d in DIGITS if y_true.count(d) > 0]

    print("\n" + "=" * 78)
    print("TABLE 5. PaddleOCR per-class classification report")
    print(f"(character-level, same-length predictions, support = {support_total:,})")
    print("=" * 78)
    print(classification_report(y_true, y_pred, labels=classes, digits=4, zero_division=0))

    print("=" * 78)
    print("SEQUENCE-LEVEL SUMMARY")
    print("=" * 78)
    print(f"Test images evaluated          : {n}")
    print(f"Exact-match (image-level)      : {exact}/{n} = {100 * exact / n:.2f}%")
    print(f"Wrong-length predictions       : {n - len(same_len)}  -> same-length subset = {len(same_len)} images")
    if support_total:
        print(f"Character-level accuracy       : {100 * sum(t == p for t, p in zip(y_true, y_pred)) / support_total:.2f}%")
    print(f"Mean confidence                : {np.mean([r[3] for r in rows]):.4f}")

    # ---- per-class stats for the plots ----
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pred_n = cm.sum(0).astype(float)
    prec = np.divide(tp, pred_n, out=np.zeros_like(tp), where=pred_n > 0)
    rec = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    char_acc = tp.sum() / support_total if support_total else 0.0

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0][0]
    vals = [100 * exact / n, 100 * char_acc]
    bars = ax.bar(["Exact\nmatch", "Character\nlevel"], vals, color=["#2563eb", "#0ea5e9"], width=0.55)
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Evaluation Summary", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[0][1]
    macro = [prec.mean(), rec.mean(), f1.mean()]
    bars = ax.bar(["Precision", "Recall", "F1"], macro, color=["#16a34a", "#f59e0b", "#7c3aed"], width=0.55)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Macro-averaged metrics", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1][0]
    bars = ax.bar(classes, f1, color="#0ea5e9", width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Digit class")
    ax.set_title("F1 Score per class", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    ax = axes[1][1]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(f"Confusion Matrix (support = {support_total:,})", fontweight="bold")

    fig.suptitle("PaddleOCR Pipeline -- Test Set Evaluation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(a.out, "paddle_classification_report.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nReport figure saved to: {out_png}")

    if not a.no_open:
        try:
            os.startfile(out_png) if sys.platform.startswith("win") else plt.show()
        except Exception as e:
            print(f"Could not open viewer ({e}); open manually: {out_png}")


if __name__ == "__main__":
    main()
