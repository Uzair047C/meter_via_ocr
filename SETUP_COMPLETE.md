# OCR App - Complete Setup

## ✅ What's Been Set Up

### Folder Structure
```
ocr_app/
├── models/
│   └── digit_only_rec/          ← Your trained recognition model
│       ├── inference.pdmodel
│       ├── inference.pdiparams
│       ├── inference.pdiparams.info
│       └── inference.yml
├── config/
│   └── config.yaml              ← Configuration file
├── utils/
│   ├── __init__.py
│   └── ocr_helper.py            ← Helper functions
├── output/                       ← Extracted text saves here
├── dataset/                      ← Place test images here
├── requirements.txt              ← Python dependencies
├── app.py                        ← Main app (interactive mode)
├── app_cli.py                    ← Command-line version
├── app_simple.py                 ← Simple version (default models)
├── README.md                     ← Documentation
└── test_app.py                   ← Testing script
```

## 📋 How to Use

### Option 1: Command-Line Mode (Simplest)
```bash
cd a:\PaddleOCR\ocr_app
python app_simple.py "path\to\image.jpg"
```

Example:
```bash
python app_simple.py "a:\PaddleOCR\PaddleOCR\paddle_digit_project\images\test\0642_04481372763crop_0.png"
```

Result: Text appears in console + saved to `output/` folder

### Option 2: With Custom Trained Model
```bash
python app_cli.py "path\to\image.jpg"
```

This uses your trained digit recognition model from `models/digit_only_rec/`

### Option 3: Interactive GUI Mode
```bash
python app.py
```

This opens a file dialog to select an image (requires display)

## 📁 Files Included

| File | Purpose |
|------|---------|
| `app.py` | Interactive app with file dialog |
| `app_cli.py` | Command-line version with custom model |
| `app_simple.py` | Simplest version using default models |
| `config/config.yaml` | Model and path configuration |
| `utils/ocr_helper.py` | Image validation, text saving |
| `models/digit_only_rec/` | Your trained recognition model |
| `output/` | Where extracted text is saved |

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run OCR on an image:**
   ```bash
   python app_simple.py "your_image.jpg"
   ```

3. **Check output:**
   - Console: Extracted text prints to screen
   - File: Saved in `output/` folder

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- `lang`: OCR language (default: 'en')
- `rec_model`: Path to recognition model
- `det_model`: Path to detection model
- Confidence thresholds and batch sizes

## 🔧 Requirements

- Python 3.7+
- paddleocr >= 2.7.0.0
- paddlepaddle-gpu >= 2.4.0
- NumPy, Pillow, OpenCV

## 📝 Example Usage

```bash
# Extract digits from a test image
python app_simple.py "a:\PaddleOCR\PaddleOCR\paddle_digit_project\images\test\0642_04481372763crop_0.png"

# Output:
# Initializing OCR...
# Processing: a:\PaddleOCR\...
# ======================================================================
# EXTRACTED TEXT:
# ======================================================================
# 
# 0642
# 04481372763
#
# ======================================================================
# ✓ Saved to: a:\PaddleOCR\ocr_app\output\0642_04481372763crop_0_output.txt
```

##  🎯 Models Used

- **Detection**: PP-LCNet_x1_0_doc_ori (default PaddleOCR)
- **Recognition**: Your custom trained model (`models/digit_only_rec/`)

## 💡 Tips

- Larger images process slower
- GPU acceleration is automatic (if available)
- Results are always saved to `output/` folder
- Check console output for confidence scores
- Use `.py` files directly from terminal with `python` command

## 🐛 Troubleshooting

**Q: No text extracted?**
- Check image quality and format
- Try a different image
- Check console for errors

**Q: Very slow processing?**
- First run downloads models (slow)
- Subsequent runs are faster
- GPU may need optimization

**Q: Can't find output?**
- Check `ocr_app/output/` folder
- Files named: `{image_name}_output.txt`

---

**App is ready to use!** 🎉
