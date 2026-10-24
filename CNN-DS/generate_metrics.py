#!/usr/bin/env python3
"""
Generate confusion matrix and metrics for CNN digit classification
Uses pre-computed data from previous runs since model loading fails due to version compatibility
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(".")
CLASSES = list(range(10))  # digits 0-9

# Load metrics from the existing JSON file
def load_existing_metrics():
    json_path = Path("A:/ResearchPaper/app/internship_computer_visions_engineering/cnns_training/runs/eval/exp/confusion_matrix_metrics.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Use test set metrics
    test_data = data['test']
    cm = np.array(test_data['confusion_matrix_counts'])
    accuracy = test_data['per_digit_accuracy']
    exact_match_acc = test_data['exact_match_accuracy']

    # Calculate precision, recall, f1, support from confusion matrix
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    # Handle division by zero
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0)
    recall = np.where((tp + fn) > 0, tp / (tp + fn), 0)
    f1 = np.where((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
    support = tp + fn  # Actual count for each class

    # Per-digit accuracy (recall)
    per_digit_acc = recall

    # Macro averages
    macro_p = np.mean(precision)
    macro_r = np.mean(recall)
    macro_f1 = np.mean(f1)

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'exact_match_acc': exact_match_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support.astype(int),
        'per_digit_acc': per_digit_acc,
        'macro_p': macro_p,
        'macro_r': macro_r,
        'macro_f1': macro_f1,
        'total_samples': test_data['num_digit_crops']
    }

def main():
    print("Loading existing metrics...")
    metrics = load_existing_metrics()

    print(f"Test Set: {metrics['total_samples']} samples ({metrics['total_samples']//5} images × 5 digits)")
    print(f"Per-Digit Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"5-Digit Exact Match: {metrics['exact_match_acc']*100:.2f}%")

    # Print metrics table
    print(f"\n{'='*60}")
    print(f"{'Digit':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Accuracy':>10}")
    print("-" * 60)
    for i, c in enumerate(CLASSES):
        print(f"{c:<8} {metrics['precision'][i]:>10.4f} {metrics['recall'][i]:>10.4f} {metrics['f1'][i]:>10.4f} {metrics['support'][i]:>10d} {metrics['per_digit_acc'][i]:>10.4f}")
    print("-" * 60)
    print(f"{'Macro':<8} {metrics['macro_p']:>10.4f} {metrics['macro_r']:>10.4f} {metrics['macro_f1']:>10.4f} {metrics['support'].sum():>10d} {metrics['accuracy']:>10.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
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
        f.write(f"Per-Digit Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"5-Digit Exact Match: {metrics['exact_match_acc']*100:.2f}%\n\n")
        f.write(f"{'Digit':<8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Accuracy':>10}\n")
        f.write("-" * 60 + "\n")
        for i, c in enumerate(CLASSES):
            f.write(f"{c:<8} {metrics['precision'][i]:>10.4f} {metrics['recall'][i]:>10.4f} {metrics['f1'][i]:>10.4f} {metrics['support'][i]:>10d} {metrics['per_digit_acc'][i]:>10.4f}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Macro':<8} {metrics['macro_p']:>10.4f} {metrics['macro_r']:>10.4f} {metrics['macro_f1']:>10.4f} {metrics['support'].sum():>10d} {metrics['accuracy']:>10.4f}\n")
        f.write(f"\n5-Digit Exact Match Accuracy: {metrics['exact_match_acc']:.4f} ({metrics['exact_match_acc']*100:.2f}%)\n")
        f.write(f"Per-Digit Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Macro Precision: {metrics['macro_p']:.4f}\n")
        f.write(f"Macro Recall: {metrics['macro_r']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")

    print(f"Saved text report to: {OUTPUT_DIR / 'metrics_report.txt'}")

    # Create and save metrics table image
    metrics_df = pd.DataFrame({
        'Digit': CLASSES,
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'Support': metrics['support'],
        'Per-Digit Accuracy': metrics['per_digit_acc']
    })
    metrics_df.loc[len(metrics_df)] = ['Macro', metrics['macro_p'], metrics['macro_r'], metrics['macro_f1'], metrics['support'].sum(), metrics['accuracy']]
    metrics_df.loc[len(metrics_df)] = ['Overall (Per-digit)', np.nan, np.nan, np.nan, np.nan, metrics['accuracy']]
    metrics_df.loc[len(metrics_df)] = ['5-Digit Exact Match', np.nan, np.nan, np.nan, np.nan, metrics['exact_match_acc']]

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