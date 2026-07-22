"""
Batch evaluation: YOLO ROI-crop -> PaddleOCR digit recognizer -> compare vs label file.

Pipeline for each entry in the label file:
  1. Load the full (uncropped) source image.
  2. Run YOLO to find + crop the meter/digit-strip ROI (highest-confidence box).
     Only the YOLO logic from your YOLO+CNN script is reused here -- the
     CNN digit-classification/segmentation code is intentionally NOT carried
     over, since PaddleOCR's recognizer replaces that step.
     If YOLO finds nothing, the image is skipped and logged separately --
     it is NOT counted as a wrong prediction, since PaddleOCR never got a
     chance to see it.
  3. Feed the cropped ROI (as an in-memory BGR array, no need to save to
     disk) into the same PaddleOCR TextRecognizer setup used in your
     standalone script.
  4. Compare the predicted string against the ground-truth label from the
     label file (exact 5-digit match + per-digit match), and accumulate
     accuracy stats.

FIX APPLIED (2026-07-21): label file entries are like
`0642_04481372763crop_0.png` but the actual files on disk are
`0642_04481372763.jpg` -- i.e. label filenames carry a `crop_0` suffix and
a `.png` extension that don't match the real `.jpg` files. Confirmed via
Get-ChildItem on ./ds/images. load_label_file() itself is untouched (it
still stores the raw label-file string in rel_path for logging/CSV output);
the normalization happens only at the point the file is actually opened,
so mismatches are still traceable back to the original label file line.

ASSUMPTIONS still worth checking before trusting the numbers this prints:
  - The `crop_0.png` -> `.jpg` transform is assumed universal across all
    212 entries. If any entries don't follow this pattern (different
    suffix, or genuinely are .png), they'll still resolve to a missing
    file and show up in skipped_missing_file -- check that list's length
    after running; if it's not 0, some entries don't fit this pattern.
  - IMAGES_ROOT is relative to the directory you run `python app.py` from,
    not relative to app.py's own location.
  - OCR confidence assumed to be a 0-1 fraction already (scaled by *100
    only for display).
"""

import os
import re
import sys
import csv
import cv2
import numpy as np
from ultralytics import YOLO

# Set up paths so python can find the ppocr package and tools (same as your
# standalone PaddleOCR script)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(CURRENT_DIR, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, CURRENT_DIR)

import utility1 as utility
from predict_rec import TextRecognizer
from ppocr.utils.utility import check_and_read

# ========================= CONFIG -- EDIT THESE =========================
YOLO_MODEL_PATH = r"./models/best.pt"
YOLO_CONF = 0.25

REC_MODEL_DIR = "./models/digit_only_rec"
REC_CHAR_DICT_PATH = "digit_dict.txt"
REC_IMAGE_SHAPE = "3,48,320"

LABEL_FILE = "./ds/labels/test_list.txt"   # path<TAB>label per line
IMAGES_ROOT = "./ds/images"   # label paths are relative to this
NUM_DIGITS = 5

RESULTS_CSV_PATH = "./output/results.csv"   # per-image predictions, for manual review
# ==========================================================================


def load_label_file(label_path):
    """Parses a PaddleOCR-style label file: one `relative_path<TAB>label`
    per line. Returns a list of (relative_path, label) tuples. Lines that
    don't split into exactly 2 tab-separated fields are logged and skipped
    rather than silently mis-parsed."""
    entries = []
    malformed = 0
    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"[label parse] line {line_num}: expected 2 tab-separated "
                      f"fields, got {len(parts)} -- skipping: {line!r}")
                malformed += 1
                continue
            rel_path, label = parts
            entries.append((rel_path.strip(), label.strip()))
    if malformed:
        print(f"[label parse] {malformed} malformed line(s) skipped in {label_path}")
    return entries


def get_meter_roi(image, yolo_model, conf=YOLO_CONF):
    """Runs YOLO on `image` (BGR array), returns (roi, confidence) for the
    highest-confidence bounding box, or None if nothing detected.
    (Same detection logic as your YOLO+CNN script's get_meter_roi -- the
    CNN-specific code from that script is intentionally not carried over.)
    """
    result = yolo_model.predict(source=image, imgsz=640, conf=conf, verbose=False)[0]

    if len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confidences))
    x1, y1, x2, y2 = boxes[best_idx]

    roi = result.orig_img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi, float(confidences[best_idx])


def build_recognizer():
    """Builds the PaddleOCR TextRecognizer exactly as your standalone
    script does, factored out so it's initialized once for the whole batch
    instead of once per image."""
    parser = utility.init_args()
    args = parser.parse_args([
        "--use_gpu=False",
        f"--rec_model_dir={REC_MODEL_DIR}",
        f"--rec_char_dict_path={REC_CHAR_DICT_PATH}",
        f"--rec_image_shape={REC_IMAGE_SHAPE}",
    ])
    return TextRecognizer(args)


def build_disk_index(images_root):
    """Scans IMAGES_ROOT once and returns {filename_stem: actual_filename}.
    Built from what's REALLY on disk, so filename resolution below matches
    reality instead of guessing a single string-transform rule that turned
    out to only hold for a subset of entries."""
    index = {}
    for fname in os.listdir(images_root):
        stem, _ext = os.path.splitext(fname)
        index[stem] = fname
    return index


def resolve_image_filename(rel_path, disk_index):
    """Tries to find the real on-disk file for a label-file entry.
    Label filenames observed to differ from disk filenames in more than
    one way (confirmed via diagnostic output, not assumed):
      - some end in `_crop_<N>` that isn't present on disk at all
      - some end in a bare `_<N>` (no "crop") that isn't present either
    Rather than picking one transform and hoping it covers every case,
    this tries progressively looser suffix strips and checks each
    candidate against the actual disk index -- only a real on-disk match
    counts as resolved."""
    stem, _ext = os.path.splitext(rel_path)

    candidates = [stem]

    no_crop_suffix = re.sub(r"_crop_\d+$", "", stem)
    if no_crop_suffix != stem:
        candidates.append(no_crop_suffix)

    no_trailing_index = re.sub(r"_\d+$", "", stem)
    if no_trailing_index != stem:
        candidates.append(no_trailing_index)

    for candidate in candidates:
        if candidate in disk_index:
            return disk_index[candidate]

    return None


def per_digit_match(pred, label):
    """Returns (num_correct_digits, num_compared). If lengths differ, the
    comparison uses the longer length as the denominator so a length
    mismatch counts against accuracy instead of being ignored."""
    n = min(len(pred), len(label))
    correct = sum(1 for i in range(n) if pred[i] == label[i])
    compared = max(len(pred), len(label))
    return correct, compared


def main():
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"Can't find YOLO weights at {YOLO_MODEL_PATH}")
    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(f"Can't find label file at {LABEL_FILE}")

    print(f"Loading YOLO from {YOLO_MODEL_PATH} ...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    print("Building PaddleOCR TextRecognizer ...")
    recognizer = build_recognizer()

    entries = load_label_file(LABEL_FILE)
    print(f"Loaded {len(entries)} labeled entries from {LABEL_FILE}")

    disk_index = build_disk_index(IMAGES_ROOT)
    print(f"Indexed {len(disk_index)} actual files in {IMAGES_ROOT}")

    total = 0
    skipped_missing_file = []
    skipped_no_roi = []
    malformed_label = []

    exact_matches = 0
    scored = 0          # entries actually scored (excludes skips/malformed)
    total_digit_correct = 0
    total_digit_compared = 0
    confidences_ocr = []
    confidences_yolo = []

    rows_for_csv = []

    for rel_path, label in entries:
        total += 1

        if len(label) != NUM_DIGITS or not label.isdigit():
            malformed_label.append((rel_path, label))
            continue

        # Resolve against what's actually on disk (handles both the
        # "_crop_N" and bare "_N" suffix conventions seen in the label file).
        resolved_filename = resolve_image_filename(rel_path, disk_index)
        if resolved_filename is None:
            skipped_missing_file.append(rel_path)
            continue
        img_path = os.path.join(IMAGES_ROOT, resolved_filename)

        img, flag, _ = check_and_read(img_path)
        if not flag:
            img = cv2.imread(img_path)
        if img is None:
            skipped_missing_file.append(rel_path)
            continue

        roi_result = get_meter_roi(img, yolo_model)
        if roi_result is None:
            skipped_no_roi.append(rel_path)
            continue
        roi, yolo_conf = roi_result

        results, _elapsed = recognizer([roi])
        pred_text, ocr_conf = results[0]

        scored += 1
        confidences_ocr.append(ocr_conf)
        confidences_yolo.append(yolo_conf)

        is_exact = (pred_text == label)
        if is_exact:
            exact_matches += 1

        correct, compared = per_digit_match(pred_text, label)
        total_digit_correct += correct
        total_digit_compared += compared

        rows_for_csv.append({
            "image": rel_path,
            "label": label,
            "prediction": pred_text,
            "exact_match": is_exact,
            "ocr_confidence": round(ocr_conf, 4),
            "yolo_confidence": round(yolo_conf, 4),
        })

        if not is_exact:
            print(f"[MISMATCH] {rel_path}: label={label} pred={pred_text} "
                  f"(ocr_conf={ocr_conf*100:.1f}%, yolo_conf={yolo_conf*100:.1f}%)")

    # ---- write per-image results for manual review ----
    if rows_for_csv:
        with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)
        print(f"\nPer-image results written to {RESULTS_CSV_PATH}")

    # ---- diagnostic: show any filenames that still have no match on disk
    # under ANY of the suffix variants tried, so a genuinely different
    # naming scheme (not just a variant we already handle) can be spotted ----
    if skipped_missing_file:
        print(f"\n[diagnostic] {len(skipped_missing_file)} still unresolved after "
              f"disk-index matching (showing up to 15) -- no match found on disk "
              f"for these under the original name, the _crop_N-stripped name, or "
              f"the bare _N-stripped name:")
        for rel_path in skipped_missing_file[:15]:
            print(f"    {rel_path}")

    # ---- summary ----
    print("\n" + "=" * 60)
    print("YOLO-ROI -> PaddleOCR evaluation summary")
    print("=" * 60)
    print(f"Total entries in label file:            {total}")
    print(f"Malformed labels (not {NUM_DIGITS} digits):        {len(malformed_label)}")
    print(f"Missing/unreadable image files:         {len(skipped_missing_file)}")
    print(f"YOLO found no ROI (skipped):             {len(skipped_no_roi)}")
    print(f"Scored (OCR actually ran):               {scored}")

    if scored > 0:
        exact_acc = exact_matches / scored * 100
        digit_acc = (total_digit_correct / total_digit_compared * 100) if total_digit_compared else 0.0
        avg_ocr_conf = float(np.mean(confidences_ocr)) * 100
        avg_yolo_conf = float(np.mean(confidences_yolo)) * 100
        print(f"\nExact 5-digit match accuracy:   {exact_acc:.2f}%  ({exact_matches}/{scored})")
        print(f"Per-digit accuracy:              {digit_acc:.2f}%")
        print(f"Average OCR confidence:          {avg_ocr_conf:.2f}%")
        print(f"Average YOLO ROI confidence:     {avg_yolo_conf:.2f}%")
    else:
        print("\nNo entries were scored -- check IMAGES_ROOT / LABEL_FILE paths "
              "and whether YOLO is detecting anything at all.")

    if skipped_no_roi:
        print(f"\n{len(skipped_no_roi)} image(s) skipped because YOLO found no ROI -- "
              f"these are NOT counted against accuracy above. Review them separately "
              f"(see console log / missing rows in {RESULTS_CSV_PATH}); reporting "
              f"exact-match accuracy without mentioning this excluded set would "
              f"overstate real-world performance.")

    print("=" * 60)


if __name__ == "__main__":
    main()