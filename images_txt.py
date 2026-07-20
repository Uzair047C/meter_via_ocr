"""
Image labeling / review tool -- writes directly into your EXISTING label
file (test_list.txt), not a new one.

For every image in IMAGES_DIR, in order:
  - If it already has a label in LABEL_FILE, that label is pre-filled in
    the text box -- press Enter to keep it as-is, or edit it first then
    press Enter to overwrite it.
  - If it has no label yet, the box is blank -- type one and press Enter
    to add it.
  - Press Escape (or click Skip) to move to the next image without
    changing anything for this one (an existing label is left untouched;
    an unlabeled image stays unlabeled).

Every Save & Next writes the label file to disk immediately (not just at
the end), so closing the window early loses nothing already saved.

MATCHING: labels are matched to images by exact filename (the first
tab-separated column in LABEL_FILE, e.g. "0642_04481372763.jpg"). Any
existing line in LABEL_FILE whose filename doesn't exactly match a file
you actually have in IMAGES_DIR (e.g. the old `..._crop_0.png` style
entries we found earlier) is left completely alone in the file -- it just
won't show up in this tool, since there's no matching image to display it
against. It will not be deleted or altered.

ASSUMPTIONS -- check before running:
  - LABEL_FILE format is `filename<TAB>label`, bare filename (no folder
    prefix) -- matches what test_list.txt actually contains for entries
    that do resolve correctly.
  - IMAGES_DIR is flat (no subfolders) -- adjust `list_images()` if not.
  - No format validation on the typed label (works for any label, not
    just 5-digit meter readings). Say the word if you want it to flag
    labels that aren't exactly 5 digits before saving.
"""

import os
import json
from collections import OrderedDict
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ========================= CONFIG -- EDIT THESE =========================
IMAGES_DIR = "./ds/images"
LABEL_FILE = "./ds/labels/test_list.txt"    # writes back into this same file
PROGRESS_FILE = "./ds/labels/.label_progress.json"   # remembers where you left off
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MAX_DISPLAY_SIZE = (700, 500)   # (width, height) cap -- large images scaled down to fit
# ==========================================================================


def load_progress(progress_path):
    """Returns the last image index the tool was showing, or 0 if there's
    no saved progress yet (first run, or the progress file was deleted)."""
    if not os.path.exists(progress_path):
        return 0
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("last_index", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        print(f"[progress] couldn't read {progress_path}, starting from image 0")
        return 0


def save_progress(progress_path, index):
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"last_index": index}, f)


def list_images(images_dir, extensions):
    images_dir = Path(images_dir)
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def load_label_file(label_path):
    """Loads LABEL_FILE into an OrderedDict {filename: label}, preserving
    original line order. Malformed lines are skipped (logged), matching
    the same tolerant parsing used elsewhere in this project."""
    labels = OrderedDict()
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"[label parse] line {line_num}: expected 2 tab-separated "
                      f"fields, got {len(parts)} -- skipping: {line!r}")
                continue
            filename, label = parts
            labels[filename.strip()] = label.strip()
    return labels


def write_label_file(label_path, labels):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for filename, label in labels.items():
            f.write(f"{filename}\t{label}\n")


class LabelingTool:
    def __init__(self, images, label_path, labels, progress_path, start_index=0):
        self.images = images
        self.label_path = label_path
        self.labels = labels   # OrderedDict, filename -> label
        self.progress_path = progress_path
        self.index = start_index

        self.root = tk.Tk()
        self.root.title("Image Label Review Tool")

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)

        self.filename_label = tk.Label(self.root, text="", font=("Segoe UI", 10))
        self.filename_label.pack()

        self.status_label = tk.Label(self.root, text="", font=("Segoe UI", 9), fg="gray")
        self.status_label.pack()

        entry_frame = tk.Frame(self.root)
        entry_frame.pack(pady=10)

        tk.Label(entry_frame, text="Label:", font=("Segoe UI", 11)).pack(side="left")
        self.entry = tk.Entry(entry_frame, font=("Segoe UI", 14), width=20)
        self.entry.pack(side="left", padx=5)
        self.entry.bind("<Return>", lambda event: self.save_and_next())
        self.entry.bind("<Escape>", lambda event: self.skip())

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        tk.Button(button_frame, text="Save & Next (Enter)", command=self.save_and_next).pack(side="left", padx=5)
        tk.Button(button_frame, text="Skip (Esc)", command=self.skip).pack(side="left", padx=5)
        tk.Button(button_frame, text="Jump to first unlabeled", command=self.jump_to_first_unlabeled).pack(side="left", padx=5)

        self.photo_ref = None  # keep a reference so PhotoImage doesn't get garbage-collected

        self.show_current_image()
        self.entry.focus_set()

    def show_current_image(self):
        if self.index >= len(self.images):
            messagebox.showinfo("Done", "That was the last image in the folder.")
            save_progress(self.progress_path, self.index)
            self.root.destroy()
            return

        save_progress(self.progress_path, self.index)

        img_path = self.images[self.index]

        pil_img = Image.open(img_path)
        pil_img.thumbnail(MAX_DISPLAY_SIZE)
        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self.photo_ref)

        self.filename_label.configure(text=img_path.name)

        existing_label = self.labels.get(img_path.name)
        already_labeled_count = sum(1 for f in self.images if f.name in self.labels)
        status = (
            f"Image {self.index + 1} of {len(self.images)}   |   "
            f"{already_labeled_count} of {len(self.images)} have a label in {os.path.basename(self.label_path)}"
        )
        self.status_label.configure(text=status)

        self.entry.delete(0, tk.END)
        if existing_label is not None:
            self.entry.insert(0, existing_label)
        self.entry.focus_set()
        self.entry.select_range(0, tk.END)  # pre-selected so typing overwrites easily if you want to edit

    def save_and_next(self):
        label = self.entry.get().strip()
        if not label:
            messagebox.showwarning("Empty label", "Type a label before saving, or click Skip.")
            return

        img_path = self.images[self.index]
        self.labels[img_path.name] = label
        write_label_file(self.label_path, self.labels)

        self.index += 1
        self.show_current_image()

    def skip(self):
        self.index += 1
        self.show_current_image()

    def jump_to_first_unlabeled(self):
        for i, img_path in enumerate(self.images):
            if img_path.name not in self.labels:
                self.index = i
                self.show_current_image()
                return
        messagebox.showinfo("Nothing left", "Every image already has a label.")

    def run(self):
        self.root.mainloop()


def main():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Can't find images directory: {IMAGES_DIR}")

    images = list_images(IMAGES_DIR, IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {IMAGES_DIR} with extensions {IMAGE_EXTENSIONS}")
        return

    labels = load_label_file(LABEL_FILE)
    already_labeled_count = sum(1 for f in images if f.name in labels)
    start_index = load_progress(PROGRESS_FILE)
    if start_index >= len(images):
        start_index = 0
    print(f"Found {len(images)} images in {IMAGES_DIR}")
    print(f"{already_labeled_count} of them already have a label in {LABEL_FILE}")
    print(f"{len(labels) - already_labeled_count} entries in {LABEL_FILE} don't match "
          f"any image currently in {IMAGES_DIR} -- left untouched, won't be shown here")
    print(f"Resuming at image {start_index + 1} of {len(images)} "
          f"(use 'Jump to first unlabeled' in the window if this isn't where you want to start)")

    tool = LabelingTool(images, LABEL_FILE, labels, PROGRESS_FILE, start_index=start_index)
    tool.run()

    final_labels = load_label_file(LABEL_FILE)
    final_count = sum(1 for f in images if f.name in final_labels)
    print(f"\nSession ended. {final_count} of {len(images)} images now have a label in {LABEL_FILE}")


if __name__ == "__main__":
    main()