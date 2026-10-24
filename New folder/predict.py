#!/usr/bin/env python3
"""
Run the segmented single-digit CNN (custom_q.keras) on a folder of 5-digit ROI
images and show, for every image, the PREDICTED reading next to the ACTUAL label.

Usage (from inside cnn-test/):
    python predict_folder.py --folder test --labels test.csv
    python predict_folder.py --folder my_new_images              # no labels -> predictions only

Arguments:
    --folder   folder containing the images (png/jpg/jpeg/bmp)
    --labels   optional CSV with columns  image,label
               (label may omit leading zeros, e.g. 3 -> 00003)
    --out      output folder (default: predictions_out)
    --per-page images per grid page in the PNG output (default: 20)

Outputs (in --out):
    results.csv           image, predicted, actual, correct, per-digit confidences
    grid_page_N.png       image grid, title = Pred / Actual, green = correct, red = wrong
    and a table + accuracy summary printed in the terminal.

Needs: tensorflow (or tensorflow-cpu), opencv-python, numpy, pandas, matplotlib
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use correct paths - model is in CNN-DS folder, test images and labels are there too
HERE = Path(__file__).resolve().parent
CNN_DS = Path("A:/ResearchPaper/app/meter_via_ocr/CNN-DS")
MODEL_PATH = CNN_DS / "custom_q.keras"
N_DIGITS = 5
DIGIT_SIZE = 32
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_model():
    import keras
    import tensorflow as tf
    import json
    import zipfile
    import os
    import tempfile
    from keras.layers import InputLayer
    from keras.models import Model

    # Custom InputLayer to handle old parameters (batch_shape, optional) not supported in newer Keras
    class CustomInputLayer(InputLayer):
        def __init__(self, **kwargs):
            # Remove problematic parameters that are no longer supported
            kwargs.pop('batch_shape', None)
            kwargs.pop('optional', None)
            # Handle input_shape if needed
            if 'input_shape' not in kwargs and 'batch_input_shape' in kwargs:
                kwargs['input_shape'] = kwargs['batch_input_shape'][1:]
                kwargs.pop('batch_input_shape')
            super().__init__(**kwargs)

    # Suppress oneDNN warnings
    tf.get_logger().setLevel('ERROR')

    # The .keras file is a zip archive - extract config and weights
    try:
        with zipfile.ZipFile(str(MODEL_PATH), 'r') as zip_file:
            # Load config.json - it has structure: {"class_name": "Functional", "config": {...}}
            config_str = zip_file.read('config.json').decode('utf-8')
            outer_config = json.loads(config_str)

            # Fix the config by removing problematic InputLayer parameters
            def fix_input_layer_config(config):
                """Recursively fix InputLayer configs to remove batch_shape and optional"""
                if isinstance(config, dict):
                    if config.get('class_name') == 'InputLayer' and 'config' in config:
                        # Remove problematic keys
                        config['config'].pop('batch_shape', None)
                        config['config'].pop('optional', None)
                        # Ensure input_shape is set
                        if 'input_shape' not in config['config'] and 'batch_input_shape' in config['config']:
                            config['config']['input_shape'] = config['config']['batch_input_shape'][1:]
                            config['config'].pop('batch_input_shape', None)
                    # Recurse into nested configs
                    for key, value in config.items():
                        if isinstance(value, (dict, list)):
                            fix_input_layer_config(value)
                elif isinstance(config, list):
                    for item in config:
                        fix_input_layer_config(item)

            fix_input_layer_config(outer_config)

            # Now load with fixed config
            with tf.keras.utils.custom_object_scope({'InputLayer': CustomInputLayer, 'Functional': Model, 'Model': Model}):
                model = keras.models.model_from_config(outer_config)

            # Extract and load weights from model.weights.h5
            weight_file = 'model.weights.h5'
            if weight_file in zip_file.namelist():
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_file.extract(weight_file, tmp_dir)
                    model.load_weights(os.path.join(tmp_dir, weight_file))
            else:
                # Try to find any h5 weight file
                for name in zip_file.namelist():
                    if name.endswith('.h5') or name.endswith('.weights.h5'):
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            zip_file.extract(name, tmp_dir)
                            model.load_weights(os.path.join(tmp_dir, name))
                        break

            return model
    except Exception as e:
        print(f"Failed to load via zip extraction: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: try with custom objects directly
        with tf.keras.utils.custom_object_scope({'InputLayer': CustomInputLayer, 'Functional': Model, 'Model': Model}):
            return keras.models.load_model(str(MODEL_PATH), compile=False)


def split_into_digits(gray, n=N_DIGITS):
    """Fixed-width slicing of the ROI into n equal vertical strips."""
    w = gray.shape[1]
    edges = np.linspace(0, w, n + 1).round().astype(int)
    return [gray[:, edges[i]:edges[i + 1]] for i in range(n)]


def prep_digit(d):
    d = cv2.resize(d, (DIGIT_SIZE, DIGIT_SIZE))
    return (d.astype(np.float32) / 255.0)[..., None]


def load_actual(labels_file):
    if not labels_file:
        return {}
    path = Path(labels_file)
    if path.suffix.lower() == '.xlsx':
        df = pd.read_excel(labels_file, dtype={"image_name": str})
        # For test.xlsx, the columns are d1, d2, d3, d4, d5 and image_name
        # We need to combine d1-d5 into a 5-digit label
        result = {}
        for _, r in df.iterrows():
            label = "".join(str(int(r[f'd{i}'])) for i in range(1, 6))
            result[str(r['image_name']).strip()] = label.zfill(N_DIGITS)
        return result
    else:
        df = pd.read_csv(labels_file, dtype={"image": str, "label": str})
        return {r["image"].strip(): str(r["label"]).strip().zfill(N_DIGITS)
                for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default="A:/ResearchPaper/app/meter_via_ocr/CNN-DS/test")
    ap.add_argument("--labels", default="A:/ResearchPaper/app/meter_via_ocr/CNN-DS/test.xlsx")
    ap.add_argument("--out", default="predictions_out")
    ap.add_argument("--per-page", type=int, default=20)
    a = ap.parse_args()

    folder, out = Path(a.folder), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images found in {folder}")
    actual = load_actual(a.labels)
    model = load_model()

    rows, images = [], []
    for p in files:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skipping unreadable image: {p.name}")
            continue
        batch = np.stack([prep_digit(d) for d in split_into_digits(gray)])
        probs = model.predict(batch, verbose=0)
        pred = "".join(str(int(i)) for i in probs.argmax(axis=1))
        conf = probs.max(axis=1)
        act = actual.get(p.name)
        rows.append({
            "image": p.name,
            "predicted": pred,
            "actual": act if act is not None else "",
            "correct": (pred == act) if act is not None else "",
            "min_digit_confidence": round(float(conf.min()), 4),
            "digit_confidences": " ".join(f"{c:.2f}" for c in conf),
        })
        images.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    # ---------------- terminal table + summary
    print(df[["image", "predicted", "actual", "correct"]].to_string(index=False))
    labeled = df[df["actual"] != ""]
    if len(labeled):
        exact = (labeled["predicted"] == labeled["actual"]).mean()
        digit_ok = sum(sum(x == y for x, y in zip(p, t))
                       for p, t in zip(labeled["predicted"], labeled["actual"]))
        digit_acc = digit_ok / (len(labeled) * N_DIGITS)
        print(f"\nLabeled images: {len(labeled)} / {len(df)}")
        print(f"5-digit exact match: {exact * 100:.2f}%   per-digit accuracy: {digit_acc * 100:.2f}%")
    else:
        print("\nNo labels supplied - showing predictions only.")

    # ---------------- image grids
    cols = 4
    for page, start in enumerate(range(0, len(rows), a.per_page), 1):
        chunk = list(range(start, min(start + a.per_page, len(rows))))
        nrows = int(np.ceil(len(chunk) / cols))
        fig, axes = plt.subplots(nrows, cols, figsize=(cols * 4.2, nrows * 2.2), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, i in zip(axes.ravel(), chunk):
            r = rows[i]
            ax.imshow(images[i])
            if r["actual"] == "":
                title, color = f"Pred: {r['predicted']}", "black"
            else:
                color = "green" if r["correct"] else "red"
                title = f"Pred: {r['predicted']}   Actual: {r['actual']}"
            ax.set_title(title, fontsize=12, fontweight="bold", color=color)
        fig.tight_layout()
        fig.savefig(out / f"grid_page_{page}.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved results.csv and grid pages to: {out.resolve()}")


if __name__ == "__main__":
    main()