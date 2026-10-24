#!/usr/bin/env python3
"""
Plot confusion matrix for CNN digit classification (not normalized)
Uses test.xlsx ground truth and model predictions
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import cv2

# Configuration
# Configuration — paths resolved relative to THIS script file
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "custom_q.keras"
TEST_XLSX = SCRIPT_DIR / "test.xlsx"
TEST_IMAGES_DIR = SCRIPT_DIR / "test"
OUTPUT_DIR = SCRIPT_DIR
CLASSES = list(range(10))  # digits 0-9

def load_model():
    import tensorflow as tf
    from keras.models import load_model
    from keras.layers import InputLayer

    # Custom InputLayer to handle old parameters
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

    # Try to handle potential version compatibility issues
    try:
        # Suppress oneDNN warnings
        tf.get_logger().setLevel('ERROR')
        return load_model(str(MODEL_PATH), compile=False,
                         custom_objects={'InputLayer': CustomInputLayer})
    except Exception as e:
        print(f"Warning: Could not load model with custom InputLayer: {e}")
        print("Falling back to standard load_model with custom objects...")
        # Last resort: try with more custom objects
        try:
            return load_model(str(MODEL_PATH), compile=False,
                            custom_objects={
                                'InputLayer': CustomInputLayer,
                                'Functional': tf.keras.models.Model,
                                'Model': tf.keras.models.Model
                            })
        except Exception as e2:
            print(f"Error: All loading methods failed: {e2}")
            raise

def preprocess_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def predict_digits(model, image_paths):
    predictions = []
    for img_path in image_paths:
        img = preprocess_image(img_path)
        if img is None:
            predictions.append(-1)
            continue
        img_batch = np.expand_dims(img, axis=0)
        pred = model.predict(img_batch, verbose=0)
        digit = np.argmax(pred, axis=1)[0]
        predictions.append(digit)
    return predictions

def main():
    print("Loading model...")
    model = load_model()
    print(f"Model loaded. Input: {model.input_shape}, Output: {model.output_shape}")

    print("Loading test labels...")
    df = pd.read_excel(TEST_XLSX)
    print(f"Loaded {len(df)} samples")

    # Ground truth: d1, d2, d3, d4, d5 columns
    y_true = df[['d1', 'd2', 'd3', 'd4', 'd5']].values.flatten()

    # Predictions
    print("Running predictions...")
    y_pred_all = []
    for _, row in df.iterrows():
        img_name = row['image_name']
        img_path = TEST_IMAGES_DIR / img_name
        if not img_path.exists():
            # try with _crop_0 suffix variants
            for suffix in ['_crop_0.png', 'crop_0.png']:
                alt = TEST_IMAGES_DIR / img_name.replace('.png', suffix)
                if alt.exists():
                    img_path = alt
                    break
        preds = predict_digits(model, [img_path] * 5)  # 5 digits per image
        y_pred_all.extend(preds)

    y_pred = np.array(y_pred_all)

    # Filter valid predictions
    mask = y_pred >= 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    print(f"Valid predictions: {len(y_true)} / {len(y_true) + (~mask).sum()}")

    # Confusion matrix (counts, not normalized)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)

    # Per-digit accuracy
    per_digit_acc = np.diag(cm) / cm.sum(axis=1)
    per_digit_acc = np.nan_to_num(per_digit_acc)

    print(f"\n{'='*60}")
    print(f"PER-DIGIT ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*60}")
    print(f"{'Digit':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Accuracy':>10}")
    print("-" * 60)
    for i, c in enumerate(CLASSES):
        print(f"{c:<8} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d} {per_digit_acc[i]:>10.4f}")
    print("-" * 60)
    macro_p = precision.mean()
    macro_r = recall.mean()
    macro_f1 = f1.mean()
    print(f"{'Macro':<8} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {support.sum():>10d} {accuracy:>10.4f}")

    # 5-digit exact match
    y_true_seq = df[['d1', 'd2', 'd3', 'd4', 'd5']].values
    y_pred_seq = y_pred.reshape(-1, 5)
    exact_matches = np.all(y_true_seq == y_pred_seq, axis=1).sum()
    exact_match_acc = exact_matches / len(y_true_seq)
    print(f"\n5-Digit Exact Match: {exact_match_acc*100:.2f}% ({exact_matches}/{len(y_true_seq)})")

    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count', 'shrink': 0.8},
                annot_kws={'size': 24, 'weight': 'bold'})  # Increased size for larger numbers
    plt.title('Confusion Matrix (Counts) - Test Set: 212 images × 5 digits = 1,060 samples',
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Predicted Digit', fontsize=16, fontweight='bold')
    plt.ylabel('True Digit', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "confusion_matrix_counts.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved confusion matrix to: {output_path}")

    # Save metrics as text report
    with open(OUTPUT_DIR / "metrics_report.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CNN DIGIT CLASSIFICATION - TEST SET METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Set: 212 images × 5 digits = 1,060 samples\n")
        f.write(f"Per-Digit Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"5-Digit Exact Match: {exact_match_acc*100:.2f}%\n\n")
        f.write(f"{'Digit':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Accuracy':>10}\n")
        f.write("-" * 60 + "\n")
        for i, c in enumerate(CLASSES):
            f.write(f"{c:<8} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10d} {per_digit_acc[i]:>10.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Macro':<8} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {support.sum():>10d} {accuracy:>10.4f}\n")
        f.write(f"\n5-Digit Exact Match Accuracy: {exact_match_acc:.4f} ({exact_match_acc*100:.2f}%)\n")
        f.write(f"Per-Digit Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Macro Precision: {macro_p:.4f}\n")
        f.write(f"Macro Recall: {macro_r:.4f}\n")
        f.write(f"Macro F1-Score: {macro_f1:.4f}\n")

    print(f"Saved text report to: {OUTPUT_DIR / 'metrics_report.txt'}")

    # Create and save metrics table image
    metrics_df = pd.DataFrame({
        'Digit': CLASSES,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support.astype(int),
        'Per-Digit Accuracy': per_digit_acc
    })
    metrics_df.loc[len(metrics_df)] = ['Macro', macro_p, macro_r, macro_f1, support.sum(), accuracy]
    metrics_df.loc[len(metrics_df)] = ['Overall (Per-digit)', np.nan, np.nan, np.nan, np.nan, accuracy]
    metrics_df.loc[len(metrics_df)] = ['5-Digit Exact Match', np.nan, np.nan, np.nan, np.nan, exact_match_acc]

    # Format DataFrame for display in table image
    display_df = metrics_df.copy()
    for col in ['Precision', 'Recall', 'F1-Score', 'Per-Digit Accuracy']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "")
    display_df['Support'] = display_df['Support'].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    table_path = OUTPUT_DIR / "metrics_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics table image to: {table_path}")

if __name__ == "__main__":
    main()