#!/usr/bin/env python3
"""
YOLOv8 meter-detector evaluation report (fix for referee point M-3).

Reports the detector properly:
  * TP / FP / FN at a stated confidence and IoU threshold
  * precision, recall, F1
  * AP@0.5 and mAP@0.5:0.95 (COCO 101-point interpolation)
  * PR curve, F1-confidence curve, precision/recall-confidence curves
  * box-level and image-level confusion matrices
  * per-image CSV + JSON summary with the thresholds and split recorded

It also runs the model on an unlabelled image folder (default ds/images) and
reports detection counts. Metrics require YOLO .txt labels; ds/ currently has
only OCR text labels, so the script automatically evaluates on the labelled
validation split of the sibling training project and says so loudly.

Usage (defaults reproduce the paper's detector evaluation):
    python yolo_detection_report.py
    python yolo_detection_report.py --conf 0.25 --iou 0.5
    python yolo_detection_report.py --images <imgs> --labels <labels>   # own split
    python yolo_detection_report.py --annotate                          # save ds overlays
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "best.pt"
DEFAULT_INFER_DIR = ROOT / "ds" / "images"
DEFAULT_LABELS_DIR = ROOT / "ds" / "labels"
# Only split in this workspace that actually ships YOLO bounding boxes:
FALLBACK_METRIC_DIR = (
    ROOT.parent / "internship_computer_visions_engineering" / "data_Set" / "validation"
)
OUT_DIR = ROOT / "output" / "yolo_detection_report"

IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
_trapz = getattr(np, "trapezoid", np.trapz)  # numpy >= 2 renamed trapz


# ----------------------------------------------------------------------------- io
def list_images(directory: Path) -> list[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(directory.glob(ext))
    return sorted(files)


def has_yolo_labels(labels_dir: Path, images_dir: Path | None = None) -> bool:
    """True when labels_dir holds per-image YOLO .txt files (not a list file)."""
    if not labels_dir.is_dir():
        return False
    if images_dir is None:
        return any(labels_dir.glob("*.txt"))
    stems = {p.stem for p in list_images(images_dir)}
    return any(p.stem in stems for p in labels_dir.glob("*.txt"))


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    """YOLO txt (cls cx cy bw bh, normalised) -> Nx4 xyxy pixel boxes."""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            boxes.append(
                [(cx - bw / 2) * w, (cy - bh / 2) * h,
                 (cx + bw / 2) * w, (cy + bh / 2) * h]
            )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = (np.clip(boxes[:, 2] - boxes[:, 0], 0, None)
              * np.clip(boxes[:, 3] - boxes[:, 1], 0, None))
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


# ------------------------------------------------------------------ prediction
def collect(model: YOLO, images: list[Path], labels_dir: Path, imgsz: int, nms_iou: float):
    """Run inference once per image at low conf; keep everything for AP sweeps."""
    records = []
    results = model.predict(
        source=[str(p) for p in images],
        conf=0.001, iou=nms_iou, imgsz=imgsz, verbose=False, stream=True,
    )
    for img_path, res in zip(images, results):
        h, w = res.orig_shape
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        keep = cls == 0  # single-class model: "meter"
        records.append({
            "path": img_path,
            "w": w, "h": h,
            "gt": load_gt(labels_dir / f"{img_path.stem}.txt", w, h),
            "boxes": xyxy[keep],
            "confs": confs[keep],
        })
    return records


# --------------------------------------------------------------------- metrics
def match_at_iou(records, iou_thr: float, conf_thr: float = 0.0):
    """Greedy per-image matching by descending confidence.

    Returns totals plus per-image dicts. Each prediction is matched to the
    highest-IoU *unmatched* ground-truth box above the IoU threshold.
    """
    total_tp = total_fp = total_fn = 0
    per_image = []
    for i, rec in enumerate(records):
        confs = rec["confs"]
        keep = confs >= conf_thr
        boxes, confs = rec["boxes"][keep], confs[keep]
        gt = rec["gt"]
        matched = np.zeros(len(gt), dtype=bool)
        tp = fp = 0
        for box, conf in sorted(zip(boxes, confs), key=lambda x: -float(x[1])):
            ious = iou_one_to_many(box, gt)
            if matched.any():
                ious[matched] = -1.0
            if len(ious) and ious.max() >= iou_thr:
                matched[int(np.argmax(ious))] = True
                tp += 1
            else:
                fp += 1
        fn = int((~matched).sum())
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_image.append({
            "image": rec["path"].name,
            "gt": len(gt),
            "pred": len(boxes),
            "tp": tp, "fp": fp, "fn": fn,
            "detected": bool(tp > 0) if len(gt) else None,
            "max_conf": float(confs.max()) if len(confs) else None,
        })
    return {"tp": total_tp, "fp": total_fp, "fn": total_fn}, per_image


def pr_arrays(records, iou_thr: float):
    """Precision/recall as the confidence threshold is swept (low -> high)."""
    n_gt = sum(len(r["gt"]) for r in records)
    preds = []
    for idx, rec in enumerate(records):
        for box, conf in zip(rec["boxes"], rec["confs"]):
            preds.append((float(conf), idx, box))
    preds.sort(key=lambda x: -x[0])

    matched = {i: np.zeros(len(r["gt"]), dtype=bool) for i, r in enumerate(records)}
    tps, fps, thresholds = [], [], []
    for conf, idx, box in preds:
        gt = records[idx]["gt"]
        ious = iou_one_to_many(box, gt)
        if matched[idx].any():
            ious[matched[idx]] = -1.0
        if len(ious) and ious.max() >= iou_thr:
            matched[idx][int(np.argmax(ious))] = True
            tps.append(1); fps.append(0)
        else:
            tps.append(0); fps.append(1)
        thresholds.append(conf)

    tps = np.asarray(tps, dtype=np.float64)
    fps = np.asarray(fps, dtype=np.float64)
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / max(n_gt, 1)
    return np.asarray(thresholds), precision, recall


def average_precision(thresholds, precision, recall) -> float:
    """AP with the same interpolation Ultralytics/COCO use (trapezoid, 101 pts)."""
    if len(thresholds) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]  # monotone envelope
    levels = np.linspace(0.0, 1.0, 101)
    return float(_trapz(np.interp(levels, mrec, mpre), levels))


def sweep_metrics(records, iou_thr: float):
    thresholds, precision, recall = pr_arrays(records, iou_thr)
    if len(thresholds) == 0:
        return {"ap": 0.0, "best_f1": 0.0, "best_conf": 0.0,
                "best_precision": 0.0, "best_recall": 0.0,
                "thresholds": thresholds, "precision": precision,
                "recall": recall, "f1": np.zeros(0)}
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    k = int(np.argmax(f1))
    return {
        "ap": average_precision(thresholds, precision, recall),
        "best_f1": float(f1[k]),
        "best_conf": float(thresholds[k]),
        "best_precision": float(precision[k]),
        "best_recall": float(recall[k]),
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def image_level_counts(records, iou_thr: float, conf_thr: float):
    """Image-level detection matrix (the form used in the paper's 150/14 claim)."""
    _, per_image = match_at_iou(records, iou_thr, conf_thr)
    tp = sum(1 for r in per_image if r["gt"] > 0 and r["detected"])
    fn = sum(1 for r in per_image if r["gt"] > 0 and not r["detected"])
    fp = sum(1 for r in per_image if r["gt"] == 0 and r["pred"] > 0)
    tn = sum(1 for r in per_image if r["gt"] == 0 and r["pred"] == 0)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


# ------------------------------------------------------------------- plotting
def plot_dashboard(records, fixed, fixed_iou, fixed_conf, sweep50, mAP50, mAP5095,
                   img_cm, out_base: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"YOLOv8 meter detection - {len(records)} images, "
        f"{fixed['tp'] + fixed['fn']} GT boxes",
        fontsize=13, fontweight="bold",
    )

    # 1. PR curve
    ax = axes[0, 0]
    th, p, r = sweep50["thresholds"], sweep50["precision"], sweep50["recall"]
    order = np.argsort(r)
    ax.plot(r[order], p[order], color="#1f77b4", lw=2,
            label=f"AP@0.5 = {sweep50['ap']:.4f}")
    ax.fill_between(r[order], 0, p[order], color="#1f77b4", alpha=0.12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall (mAP@0.5 = {mAP50:.4f}, "
                 f"mAP@0.5:0.95 = {mAP5095:.4f})")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")

    # 2. F1 vs confidence
    ax = axes[0, 1]
    ax.plot(th, sweep50["f1"], color="#2ca02c", lw=2)
    ax.scatter([sweep50["best_conf"]], [sweep50["best_f1"]], color="#d62728", zorder=3)
    ax.annotate(f"best F1={sweep50['best_f1']:.4f}\n@ conf={sweep50['best_conf']:.3f}",
                (sweep50["best_conf"], sweep50["best_f1"]),
                textcoords="offset points", xytext=(10, -25), fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("F1")
    ax.set_title("F1 vs confidence (IoU >= 0.50)")
    ax.grid(alpha=0.3)

    # 3. Precision & recall vs confidence
    ax = axes[1, 0]
    ax.plot(th, sweep50["precision"], color="#1f77b4", lw=2, label="Precision")
    ax.plot(th, sweep50["recall"], color="#ff7f0e", lw=2, label="Recall")
    ax.axvline(fixed_conf, color="grey", ls="--", lw=1,
               label=f"reported conf = {fixed_conf:.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs confidence (IoU >= 0.50)")
    ax.grid(alpha=0.3); ax.legend(loc="lower left")

    # 4. Box-level confusion matrix
    ax = axes[1, 1]
    tp, fp, fn = fixed["tp"], fixed["fp"], fixed["fn"]
    cm = np.array([[tp, fn], [fp, 0]])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(tp, fp, fn, 1))
    for (row, col), val in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [tp, fn, fp, None],
    ):
        ax.text(col, row, str(val) if val is not None else "-",
                ha="center", va="center", fontsize=18,
                color="white" if val and val > (cm.max() / 2) else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred meter", "Pred background"])
    ax.set_yticklabels(["Actual meter", "Actual background"])
    ax.set_title(
        f"Box-level confusion matrix\n"
        f"conf >= {fixed_conf:.2f}, IoU >= {fixed_iou:.2f}  |  "
        f"P={tp / max(tp + fp, 1):.4f}  R={tp / max(tp + fn, 1):.4f}  "
        f"F1={2 * tp / max(2 * tp + fp + fn, 1):.4f}\n"
        f"image-level: TP={img_cm['tp']} FN={img_cm['fn']} "
        f"FP={img_cm['fp']} TN={img_cm['tn']}"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--images", type=Path, default=None,
                    help="labelled image folder for metrics")
    ap.add_argument("--labels", type=Path, default=None,
                    help="YOLO label folder matching --images")
    ap.add_argument("--infer-dir", type=Path, default=DEFAULT_INFER_DIR,
                    help="unlabelled folder to run the detector on (default: ds/images)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold for the reported confusion matrix")
    ap.add_argument("--iou", type=float, default=0.50,
                    help="IoU threshold for a true positive in the reported matrix")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--nms-iou", type=float, default=0.70)
    ap.add_argument("--annotate", action="store_true",
                    help="save annotated predictions for --infer-dir")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    # ---- pick the metric split -------------------------------------------------
    images_dir, labels_dir = args.images, args.labels
    if images_dir is None or labels_dir is None:
        if has_yolo_labels(DEFAULT_LABELS_DIR, DEFAULT_INFER_DIR):
            images_dir = DEFAULT_INFER_DIR
            labels_dir = DEFAULT_LABELS_DIR
        else:
            images_dir = FALLBACK_METRIC_DIR / "images"
            labels_dir = FALLBACK_METRIC_DIR / "labels"
            print("[NOTE] ds/labels contains no YOLO .txt boxes (OCR labels only).")
            print(f"[NOTE] Detection metrics are therefore computed on the labelled split:")
            print(f"       {images_dir.parent}")
            print("[NOTE] Run inference on ds/images is reported separately below.\n")
    if not images_dir.is_dir():
        raise SystemExit(f"image folder not found: {images_dir}")
    if not has_yolo_labels(labels_dir, images_dir):
        raise SystemExit(f"no YOLO .txt labels for the images in {images_dir} "
                         f"(label folder: {labels_dir})")

    metric_images = list_images(images_dir)
    if not metric_images:
        raise SystemExit(f"no images in: {images_dir}")

    model = YOLO(str(args.model))
    records = collect(model, metric_images, labels_dir, args.imgsz, args.nms_iou)

    # ---- metrics ---------------------------------------------------------------
    fixed, per_image = match_at_iou(records, args.iou, args.conf)
    tp, fp, fn = fixed["tp"], fixed["fp"], fixed["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    sweep50 = sweep_metrics(records, 0.5)
    aps = [sweep_metrics(records, t)["ap"]
           for t in np.arange(0.5, 0.951, 0.05)]
    map50 = float(aps[0])
    map5095 = float(np.mean(aps))

    img_cm = image_level_counts(records, args.iou, args.conf)
    n_gt = sum(len(r["gt"]) for r in records)
    n_pred = tp + fp

    # ---- console report --------------------------------------------------------
    print("=" * 66)
    print("YOLO DETECTION RESULTS")
    print("=" * 66)
    print(f"model            : {args.model}")
    print(f"evaluation split : {images_dir.parent}")
    print(f"images           : {len(records)}")
    print(f"GT boxes         : {n_gt}")
    print(f"thresholds       : conf >= {args.conf:.2f}, "
          f"TP IoU >= {args.iou:.2f}, NMS IoU = {args.nms_iou:.2f}, "
          f"imgsz = {args.imgsz}")
    print("-" * 66)
    print(f"TP = {tp}   FP = {fp}   FN = {fn}   (TN undefined for detection)")
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1        = {f1:.4f}")
    print(f"mAP@0.5      = {map50:.4f}")
    print(f"mAP@0.5:0.95 = {map5095:.4f}  "
          f"(AP@0.75 = {aps[5]:.4f}, AP@0.90 = {aps[8]:.4f})")
    print(f"best F1   = {sweep50['best_f1']:.4f} at conf = "
          f"{sweep50['best_conf']:.4f} "
          f"(P={sweep50['best_precision']:.4f}, R={sweep50['best_recall']:.4f})")
    print("-" * 66)
    print("Count reconciliation")
    print(f"  GT boxes        = TP + FN = {tp} + {fn} = {tp + fn}")
    print(f"  Pred boxes      = TP + FP = {tp} + {fp} = {n_pred}")
    print(f"  unmatched preds = FP       = {fp}")
    print(f"  missed GT       = FN       = {fn}")
    print(f"  image-level     = TP {img_cm['tp']}  FN {img_cm['fn']}  "
          f"FP {img_cm['fp']}  TN {img_cm['tn']}  "
          f"(sum = {sum(img_cm.values())} images)")
    print("=" * 66)

    # ---- outputs ---------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dashboard(records, fixed, args.iou, args.conf, sweep50,
                   map50, map5095, img_cm, OUT_DIR / "yolo_detection_report")

    import csv
    with open(OUT_DIR / "yolo_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    summary = {
        "model": str(args.model),
        "evaluation_split": str(images_dir.parent),
        "images": len(records),
        "gt_boxes": n_gt,
        "thresholds": {
            "confidence": args.conf,
            "match_iou": args.iou,
            "nms_iou": args.nms_iou,
            "imgsz": args.imgsz,
        },
        "box_level": {"tp": tp, "fp": fp, "fn": fn},
        "image_level": img_cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": map50,
        "mAP50_95": map5095,
        "ap_per_iou": {f"{t:.2f}": a for t, a in zip(np.arange(0.5, 0.951, 0.05), aps)},
        "best_f1": sweep50["best_f1"],
        "best_f1_confidence": sweep50["best_conf"],
        "best_f1_precision": sweep50["best_precision"],
        "best_f1_recall": sweep50["best_recall"],
    }

    # ---- unlabelled inference on ds -------------------------------------------
    if args.infer_dir.is_dir():
        infer_images = list_images(args.infer_dir)
        print(f"\nInference on unlabelled folder: {args.infer_dir}")
        print(f"  images: {len(infer_images)}")
        pred_results = model.predict(
            source=[str(p) for p in infer_images],
            conf=args.conf, iou=args.nms_iou, imgsz=args.imgsz,
            verbose=False, stream=True,
        )
        detected = 0
        total_boxes = 0
        confs = []
        rows = []
        anno_dir = OUT_DIR / "annotated"
        if args.annotate:
            anno_dir.mkdir(parents=True, exist_ok=True)
        for img_path, res in zip(infer_images, pred_results):
            c = res.boxes.conf.cpu().numpy()
            detected += int(len(c) > 0)
            total_boxes += len(c)
            confs.extend(c.tolist())
            rows.append({"image": img_path.name, "pred_boxes": len(c),
                         "max_conf": float(c.max()) if len(c) else "",
                         "mean_conf": float(c.mean()) if len(c) else ""})
            if args.annotate:
                res.save(str(anno_dir / img_path.name))
        print(f"  images with >=1 detection: {detected} / {len(infer_images)}")
        print(f"  predicted boxes           : {total_boxes}")
        if confs:
            print(f"  confidence                : mean={np.mean(confs):.4f}  "
                  f"min={np.min(confs):.4f}  max={np.max(confs):.4f}")
        with open(OUT_DIR / "yolo_infer_predictions.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        summary["unlabelled_inference"] = {
            "folder": str(args.infer_dir),
            "images": len(infer_images),
            "images_detected": detected,
            "predicted_boxes": total_boxes,
            "note": "no ground-truth boxes available; detection counts only",
        }
        if args.annotate:
            print(f"  annotated images          : {anno_dir}")

    with open(OUT_DIR / "yolo_metrics_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nSaved: {OUT_DIR / 'yolo_detection_report.png'}")
    print(f"       {OUT_DIR / 'yolo_per_image.csv'}")
    print(f"       {OUT_DIR / 'yolo_metrics_summary.json'}")


if __name__ == "__main__":
    main()
