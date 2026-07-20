# PaddleOCR Application

A complete, organized OCR application using PaddleOCR with **custom trained recognition model**.

## Folder Structure

```
ocr_app/
├── models/              # Pre-trained & custom trained models
│   └── digit_only_rec/  # Your trained recognition model
├── dataset/             # Test images and training data
├── output/              # Extracted text outputs
├── config/              # Configuration files
├── utils/               # Helper functions
├── app.py               # Main application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Models Included

- **digit_only_rec/** - Your custom trained recognition model
  - Files: `inference.pdmodel`, `inference.pdiparams`, `inference.yml`

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Select an image from the dialog

## Files Description

- **models/digit_only_rec/** - Your trained recognition weights
- **dataset/** - Place test images here for batch processing
- **output/** - Extracted text files are saved here
- **config/** - Configuration settings (YAML)
- **utils/** - Helper functions for image validation and file handling
- **app.py** - Main application with GUI file picker and custom model loading
- **requirements.txt** - All Python package dependencies

## Usage

Run the app and select an image. Text is extracted using your trained model and saved to the output folder.

