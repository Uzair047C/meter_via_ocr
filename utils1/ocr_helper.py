"""OCR Helper Functions"""
from pathlib import Path


def validate_image(image_path):
    """Validate if file is a valid image."""
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    return Path(image_path).suffix.lower() in valid_formats


def save_text_output(text, output_path):
    """Save extracted text to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


def get_image_files(directory):
    """Get all image files from directory."""
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    return [
        str(f) for f in Path(directory).rglob('*')
        if f.suffix.lower() in valid_formats
    ]
