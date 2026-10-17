#!/usr/bin/env python3
"""
YOLO Confusion Matrix for Meter Detection
- If ds/labels exists (YOLO txt format): IoU-matched detection confusion matrix (TP/FP/FN).
- Otherwise: image-level fallback (every image assumed to contain a meter).
Saves the confusion matrix as PNG and opens it in the default viewer.
"""
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = Path(r"A:\ResearchPaper\app\meter_via_ocr\models\best.pt")
IMAGES_DIR = Path(r"A:\ResearchPaper\app\meter_via_ocr\ds\images")
LABELS_DIR = IMAGES_DIR.parent / "labels"          # YOLO txt labels
OUTPUT_PNG = Path(r"A:\ResearchPaper\app\meter_via_ocr\confusion_matrix.png")

CONF_THRESH = 0.25   # prediction confidence threshold
IOU_THRESH = 0.5     # IoU needed to count a prediction as a true positive
METER_CLASS = 0      # class id of "meter"


def display_image(path: Path):
    """Open the saved image in the OS default viewer."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as e:
        print(f"Could not open viewer ({e}); open manually: {path}")


def iou(a, b):
    """IoU of two boxes in xyxy format."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
    """Read YOLO txt (cls cx cy w h, normalized) -> list of xyxy pixel boxes for METER_CLASS."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != METER_CLASS:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append([(cx - w / 2) * img_w, (cy - h / 2) * img_h,
                      (cx + w / 2) * img_w, (cy + h / 2) * img_h])
    return boxes


def match_image(pred_boxes, pred_confs, gt_boxes):
    """Greedy IoU matching (highest confidence first). Returns tp, fp, fn."""
    order = np.argsort(-np.array(pred_confs)) if len(pred_confs) else []
    matched = set()
    tp = fp = 0
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if j in matched:
                continue
            v = iou(pred_boxes[i], g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= IOU_THRESH:
            matched.add(best_j)
            tp += 1
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def plot_matrix(cm, annot, title, path):
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Meter', 'Background'],
                yticklabels=['Meter', 'Background'],
                cbar_kws={'label': 'Count'})
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    model = YOLO(str(MODEL_PATH))

    image_files = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
    print(f"Found {len(image_files)} images")
    use_labels = LABELS_DIR.exists()
    if not use_labels:
        print(f"[WARN] No labels folder at {LABELS_DIR}. Falling back to image-level mode "
              f"(all images assumed to contain a meter) - precision/F1 are NOT meaningful.")

    TP = FP = FN = 0
    img_detected = 0
    confs_all = []

    for img_path in image_files:
        res = model(str(img_path), conf=CONF_THRESH, verbose=False)[0]
        h, w = res.orig_shape

        cls = res.boxes.cls.cpu().numpy().astype(int)
        keep = cls == METER_CLASS
        pred_boxes = res.boxes.xyxy.cpu().numpy()[keep].tolist()
        pred_confs = res.boxes.conf.cpu().numpy()[keep].tolist()
        confs_all.extend(pred_confs)
        img_detected += int(len(pred_boxes) > 0)

        if use_labels:
            gt = load_gt_boxes(LABELS_DIR / f"{img_path.stem}.txt", w, h)
            tp, fp, fn = match_image(pred_boxes, pred_confs, gt)
            TP += tp; FP += fp; FN += fn

    n = len(image_files)
    print("\n" + "=" * 50)

    if use_labels:
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        print(f"DETECTION CONFUSION MATRIX (IoU >= {IOU_THRESH}, conf >= {CONF_THRESH})")
        print("=" * 50)
        print(f"TP: {TP}   FP: {FP}   FN: {FN}   (TN undefined for detection)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        cm = np.array([[TP, FN], [FP, 0]])
        annot = np.array([[str(TP), str(FN)], [str(FP), "-"]])
        title = (f"YOLO Meter Detection (IoU>={IOU_THRESH}, conf>={CONF_THRESH})\n"
                 f"P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
    else:
        recall = img_detected / n if n else 0.0
        print("IMAGE-LEVEL DETECTION RATE (no labels available)")
        print("=" * 50)
        print(f"Detected: {img_detected} / {n} ({recall * 100:.1f}%)")
        print(f"Missed:   {n - img_detected} / {n} ({(1 - recall) * 100:.1f}%)")

        cm = np.array([[img_detected, n - img_detected], [0, 0]])
        annot = np.array([[str(img_detected), str(n - img_detected)], ["-", "-"]])
        title = f"YOLO Meter Detection (image-level)\nDetection rate: {recall:.2%}"

    if confs_all:
        print(f"\nDetection confidence: mean={np.mean(confs_all):.4f}  "
              f"min={np.min(confs_all):.4f}  max={np.max(confs_all):.4f}  "
              f"std={np.std(confs_all):.4f}")

    plot_matrix(cm, annot, title, OUTPUT_PNG)
    print(f"\nSaved confusion matrix to: {OUTPUT_PNG}")
    display_image(OUTPUT_PNG)


if __name__ == "__main__":
    main()