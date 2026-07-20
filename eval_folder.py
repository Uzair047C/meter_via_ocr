import os
import sys
import time
import cv2
import numpy as np

# Set up paths so python can find the ppocr package and tools
current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(current_dir, "tools")
sys.path.insert(0, tools_dir)
sys.path.insert(0, current_dir)

import utility1 as utility
from predict_rec import TextRecognizer
from ppocr.utils.utility import check_and_read

def load_labels(label_file_path):
    """Loads image filename -> ground truth label mapping from a label file."""
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
                # Get the basename of the image path to match filenames correctly
                img_name = os.path.basename(parts[0].strip())
                label = parts[1].strip()
                labels[img_name] = label
    return labels

def main():
    # Set default paths relative to ocr_app
    default_image_dir = os.path.join(current_dir, "dataset", "images", "test")
    default_label_file = os.path.join(current_dir, "dataset", "labels", "test_list.txt")
    
    print("=" * 60)
    print("PaddleOCR Folder Evaluation Script")
    print("=" * 60)
    
    # Allow user overrides (pressing Enter accepts defaults)
    image_dir = input(f"Enter image folder path [Default: {default_image_dir}]: ").strip()
    if not image_dir:
        image_dir = default_image_dir
        
    label_file = input(f"Enter label file path [Default: {default_label_file}]: ").strip()
    if not label_file:
        label_file = default_label_file
        
    if not os.path.exists(image_dir):
        print(f"Error: Image directory '{image_dir}' does not exist.")
        return
        
    # Load labels
    labels = load_labels(label_file)
    if not labels:
        print("Continuing without ground-truth labels (will only output predictions).")
    else:
        print(f"Loaded {len(labels)} ground-truth labels.")
        
    # Find all image files
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_formats
    ]
    
    if not image_files:
        print(f"Error: No valid images found in {image_dir}")
        return
        
    print(f"Found {len(image_files)} images to evaluate.")
    
    # Initialize the OCR recognizer
    parser = utility.init_args()
    args = parser.parse_args([
        "--use_gpu=False",
        f"--image_dir={image_dir}",
        "--rec_model_dir=./models/digit_only_rec",
        "--rec_char_dict_path=./dataset/digit_dict.txt",
        "--rec_image_shape=3,48,320"
    ])
    
    print("\nInitializing OCR Engine (loading model)...")
    recognizer = TextRecognizer(args)
    
    # Evaluation metrics
    correct_count = 0
    total_evaluated = 0
    total_chars_correct = 0
    total_chars_possible = 0
    total_confidence = 0.0
    
    results_summary = []
    failures = []
    
    print("Running predictions on image folder...")
    start_time = time.time()
    
    # Process images in batches for efficiency
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
            
        # Run recognition on the batch
        batch_results, _ = recognizer(img_list)
        
        for fpath, (pred_text, confidence) in zip(valid_batch_files, batch_results):
            filename = os.path.basename(fpath)
            gt_label = labels.get(filename, None)
            
            total_evaluated += 1
            total_confidence += confidence
            
            is_correct = None
            char_acc = None
            
            if gt_label is not None:
                is_correct = (pred_text == gt_label)
                if is_correct:
                    correct_count += 1
                
                # Calculate character-level accuracy (position matching)
                matches = sum(1 for c1, c2 in zip(pred_text, gt_label) if c1 == c2)
                total_chars_correct += matches
                total_chars_possible += max(len(pred_text), len(gt_label))
                char_acc = matches / max(len(pred_text), len(gt_label))
                
                if not is_correct:
                    failures.append({
                        'filename': filename,
                        'gt': gt_label,
                        'pred': pred_text,
                        'conf': confidence
                    })
                    
            results_summary.append({
                'filename': filename,
                'gt': gt_label,
                'pred': pred_text,
                'conf': confidence,
                'correct': is_correct,
                'char_acc': char_acc
            })
            
    elapsed_time = time.time() - start_time
    
    # Print results summary to console
    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    print(f"Total Evaluated Images: {total_evaluated}")
    print(f"Total Time Taken:       {elapsed_time:.2f} seconds")
    print(f"Average Speed:          {elapsed_time / total_evaluated:.4f} seconds/image")
    print(f"Average Confidence:     {total_confidence / total_evaluated * 100:.2f}%")
    
    if labels:
        exact_match_acc = (correct_count / total_evaluated) * 100
        char_level_acc = (total_chars_correct / total_chars_possible) * 100 if total_chars_possible > 0 else 0.0
        print(f"Exact Match Accuracy:   {exact_match_acc:.2f}% ({correct_count}/{total_evaluated})")
        print(f"Char-level Accuracy:    {char_level_acc:.2f}% ({total_chars_correct}/{total_chars_possible} digits)")
        
        # Save a detailed text report
        report_dir = os.path.join(current_dir, "output")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "evaluation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as rf:
            rf.write("=" * 75 + "\n")
            rf.write("PaddleOCR Evaluation Report\n")
            rf.write("=" * 75 + "\n")
            rf.write(f"Total Evaluated Images: {total_evaluated}\n")
            rf.write(f"Total Execution Time:   {elapsed_time:.2f} seconds\n")
            rf.write(f"Exact Match Accuracy:   {exact_match_acc:.2f}% ({correct_count}/{total_evaluated})\n")
            rf.write(f"Char-level Accuracy:    {char_level_acc:.2f}%\n")
            rf.write(f"Average Confidence:     {total_confidence / total_evaluated * 100:.2f}%\n")
            rf.write("=" * 75 + "\n\n")
            
            if failures:
                rf.write("Failed Classifications (Misidentified):\n")
                rf.write("-" * 80 + "\n")
                rf.write(f"{'Image Filename':<40} | {'Ground Truth':<12} | {'Predicted':<12} | {'Confidence':<10}\n")
                rf.write("-" * 80 + "\n")
                for f in sorted(failures, key=lambda x: x['filename']):
                    rf.write(f"{f['filename']:<40} | {f['gt']:<12} | {f['pred']:<12} | {f['conf']*100:.2f}%\n")
                rf.write("-" * 80 + "\n")
            else:
                rf.write("All images matched perfectly!\n")
                
        print(f"\n✓ Detailed report saved to: {report_path}")
        
        # Show top failures in console
        if failures:
            print("\nTop Misclassified Examples (first 5):")
            print("-" * 75)
            print(f"{'Image Filename':<35} | {'Ground Truth':<12} | {'Predicted':<12} | {'Confidence':<10}")
            print("-" * 75)
            for f in failures[:5]:
                print(f"{f['filename']:<35} | {f['gt']:<12} | {f['pred']:<12} | {f['conf']*100:.2f}%")
            print("-" * 75)
    else:
        # Save predictions list if no ground truth was provided
        report_dir = os.path.join(current_dir, "output")
        os.makedirs(report_dir, exist_ok=True)
        predictions_path = os.path.join(report_dir, "predictions.txt")
        with open(predictions_path, 'w', encoding='utf-8') as pf:
            for item in results_summary:
                pf.write(f"{item['filename']}\t{item['pred']}\t{item['conf']:.4f}\n")
        print(f"\n✓ Predictions list saved to: {predictions_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
