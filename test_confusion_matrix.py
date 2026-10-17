#!/usr/bin/env python3
"""
Test set confusion matrix for meter_via_ocr project.
Loads test images, runs OCR predictions, and generates confusion matrix.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import sys
import cv2
from pathlib import Path

# Add tools directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(current_dir, "tools")
sys.path.insert(0, tools_dir)
sys.path.insert(0, current_dir)

import utility1 as utility
from predict_rec import TextRecognizer
from ppocr.utils.utility import check_and_read

def load_labels(label_file_path):
    """Load image filename -> ground truth label mapping from label file."""
    labels = {}
    if not os.path.exists(label_file_path):
        print(f"Warning: Label file {label_file_path} not found.")
        return labels

    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                img_name = os.path.basename(parts[0].strip())
                label = parts[1].strip()
                labels[img_name] = label
    return labels

def main():
    print("=" * 70)
    print("METER VIA OCR - TEST SET CONFUSION MATRIX")
    print("=" * 70)

    # Configuration
    IMAGE_DIR = os.path.join(current_dir, "dataset", "images", "test")
    LABEL_FILE = os.path.join(current_dir, "dataset", "labels", "test_list.txt")
    MODEL_DIR = os.path.join(current_dir, "models", "digit_only_rec")
    CHAR_DICT_PATH = os.path.join(current_dir, "digit_dict.txt")

    print(f"Image directory: {IMAGE_DIR}")
    print(f"Label file: {LABEL_FILE}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Character dictionary: {CHAR_DICT_PATH}")

    # Load test labels
    print("\nLoading test labels...")
    test_labels = load_labels(LABEL_FILE)
    print(f"Loaded {len(test_labels)} test labels.")

    # Show first few labels as sample
    if test_labels:
        print("\nSample labels:")
        for i, (fname, label) in enumerate(list(test_labels.items())[:5]):
            print(f"  {fname}: {label}")

    # Initialize OCR model
    print("\nInitializing OCR model...")
    parser = utility.init_args()
    args = parser.parse_args([
        "--use_gpu=False",
        f"--image_dir={IMAGE_DIR}",
        "--rec_model_dir=./models/digit_only_rec",
        f"--rec_char_dict_path={CHAR_DICT_PATH}",
        "--rec_image_shape=3,48,320"
    ])

    recognizer = TextRecognizer(args)
    print("Model loaded successfully.")

    # Get all test image files
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
        if os.path.splitext(f)[1].lower() in valid_formats
    ]

    print(f"\nFound {len(image_files)} test images to evaluate.")

    # Filter to only those with labels available
    labeled_images = []
    missing_labels = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        if filename in test_labels:
            labeled_images.append(img_path)
        else:
            missing_labels.append(filename)

    print(f"Images with labels: {len(labeled_images)}")
    if missing_labels:
        print(f"Images missing labels: {len(missing_labels)} (first 5: {missing_labels[:5]})")

    # Use only labeled images for evaluation
    image_files = labeled_images

    # Run OCR predictions
    print("\nRunning OCR predictions on test images...")

    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels
    results = []  # Detailed results
    confidences = []

    # Process in batches
    batch_size = 16
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        img_list = []
        valid_batch_files = []

        for fpath in batch_files:
            img, flag, _ = check_and_read(fpath)
            if not flag:
                img = cv2.imread(fpath)
            if img is not None:
                img_list.append(img)
                valid_batch_files.append(fpath)
            else:
                print(f"Warning: Could not load image {fpath}")

        if not img_list:
            continue

        # Run recognition on batch
        batch_results, _ = recognizer(img_list)

        for fpath, (pred_text, confidence) in zip(valid_batch_files, batch_results):
            filename = os.path.basename(fpath)
            gt_label = test_labels.get(filename)

            if gt_label is not None:
                y_true.append(gt_label)
                y_pred.append(pred_text)
                confidences.append(confidence)
                results.append({
                    'filename': filename,
                    'ground_truth': gt_label,
                    'prediction': pred_text,
                    'confidence': confidence,
                    'correct': gt_label == pred_text
                })

    print(f"Processed {len(y_true)} images with ground truth labels.")

    # Calculate overall accuracy
    if y_true:
        exact_matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = exact_matches / len(y_true)
        print(f"Exact match accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Exact matches: {exact_matches} / {len(y_true)}")
        print(f"Average confidence: {np.mean(confidences):.4f} ({np.mean(confidences)*100:.2f}%)")

    # Prepare data for confusion matrix (character-level)
    print("\nPreparing confusion matrix data (character-level)...")

    # Flatten the sequences to get individual character predictions
    true_chars = []
    pred_chars = []

    for true_seq, pred_seq in zip(y_true, y_pred):
        # Pad shorter sequence with a special character or truncate longer one
        max_len = max(len(true_seq), len(pred_seq))
        true_padded = true_seq.ljust(max_len, ' ')[:max_len]
        pred_padded = pred_seq.ljust(max_len, ' ')[:max_len]

        for t_char, p_char in zip(true_padded, pred_padded):
            true_chars.append(t_char)
            pred_chars.append(p_char)

    print(f"Total character predictions: {len(true_chars)}")

    # Get unique characters (digits 0-9 plus space for padding)
    unique_chars = sorted(list(set(true_chars + pred_chars)))
    print(f"Unique characters found: {unique_chars}")

    # Create mapping from char to index for confusion matrix
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}

    # Convert character sequences to indices
    y_true_indices = [char_to_idx[c] for c in true_chars]
    y_pred_indices = [char_to_idx[c] for c in pred_chars]

    # Generate confusion matrix
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(len(unique_chars))))
    print(f"Confusion matrix shape: {cm.shape}")

    # Also compute normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(f"Normalized confusion matrix shape: {cm_norm.shape}")

    # Plot confusion matrices
    print("\nGenerating confusion matrix plots...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot confusion matrix (counts)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=unique_chars, yticklabels=unique_chars)
    axes[0].set_title('Confusion Matrix (Counts)\nCharacter-level predictions on test set')
    axes[0].set_xlabel('Predicted Character')
    axes[0].set_ylabel('True Character')

    # Plot normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
                xticklabels=unique_chars, yticklabels=unique_chars, vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted Character')
    axes[1].set_ylabel('True Character')

    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_confusion_matrix.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {output_path}")

    # Show the plot
    plt.show()

    # Print detailed metrics
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORT (Character-level)")
    print("=" * 70)

    if y_true_indices and y_pred_indices:
        report = classification_report(y_true_indices, y_pred_indices, target_names=unique_chars)
        print(report)

    # Also show sequence-level metrics
    print("\n" + "=" * 70)
    print("SEQUENCE-LEVEL METRICS")
    print("=" * 70)

    if y_true and y_pred:
        # Exact match accuracy
        exact_matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        exact_match_accuracy = exact_matches / len(y_true)
        print(f"Exact Match Accuracy: {exact_match_accuracy:.4f} ({exact_match_accuracy*100:.2f}%)")
        print(f"Exact matches: {exact_matches} / {len(y_true)}")

        # Character-level accuracy (position-based)
        total_chars = sum(len(t) for t in y_true)
        correct_chars = sum(sum(1 for tc, pc in zip(t, p) if tc == pc) for t, p in zip(y_true, y_pred))
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        print(f"Character-level Accuracy: {char_accuracy:.4f} ({char_accuracy*100:.2f}%)")
        print(f"Correct characters: {correct_chars} / {total_chars}")

        # Show some examples of failures
        failures = [r for r in results if not r['correct']]
        if failures:
            print(f"\nNumber of failed predictions: {len(failures)} out of {len(results)}")
            print("\nFirst 10 failures:")
            for i, failure in enumerate(failures[:10]):
                print(f"  {i+1}. {failure['filename']}: expected '{failure['ground_truth']}', got '{failure['prediction']}' (conf: {failure['confidence']:.3f})")
        else:
            print("\nAll predictions were correct!")

    # Save results to JSON for further analysis
    results_dir = os.path.join(current_dir, "output")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "test_results.json")

    results_data = {
        'summary': {
            'total_images': len(results),
            'exact_matches': sum(1 for r in results if r['correct']),
            'exact_match_accuracy': sum(1 for r in results if r['correct']) / len(results) if results else 0,
            'average_confidence': np.mean([r['confidence'] for r in results]) if results else 0
        },
        'confusion_matrix': {
            'matrix': cm.tolist(),
            'normalized': cm_norm.tolist(),
            'labels': unique_chars
        },
        'detailed_results': results
    }

    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nDetailed results saved to: {results_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()