import csv
import os
from tkinter import Tk, filedialog

# Select CSV file
Tk().withdraw()
csv_path = filedialog.askopenfilename(
    title="Select CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

if not csv_path:
    print("No file selected.")
    exit()

# Output TXT path (same name as CSV)
txt_path = os.path.splitext(csv_path)[0] + ".txt"

with open(csv_path, "r", newline="", encoding="utf-8") as csv_file, \
     open(txt_path, "w", encoding="utf-8") as txt_file:

    reader = csv.reader(csv_file)

    # Skip header
    next(reader, None)

    for row in reader:
        if len(row) < 2:
            continue

        image_name = row[0].strip()
        label = row[1].strip().zfill(5)  # Pad to 5 digits

        txt_file.write(f"{image_name}\t{label}\n")

print(f"Done!\nTXT saved as:\n{txt_path}")