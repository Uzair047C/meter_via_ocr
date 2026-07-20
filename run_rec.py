import os
import sys
import cv2

# Set up paths so python can find the ppocr package and tools
current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(current_dir, "tools")
sys.path.insert(0, tools_dir)
sys.path.insert(0, current_dir)

import utility1 as utility
from predict_rec import TextRecognizer
from ppocr.utils.utility import check_and_read

def select_image_file():
    """Opens a file dialog to select an image, with console input fallback."""
    try:
        from tkinter import filedialog, Tk
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        print("Opening file selection dialog...")
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        if file_path:
            return file_path
    except Exception as e:
        print(f"GUI Dialog could not be opened: {e}")
    
    # Fallback to console input
    path = input("Enter the path to the image file: ").strip()
    if path.startswith(('"', "'")) and path.endswith(('"', "'")):
        path = path[1:-1]
    return path

def main():
    # Ask the user to choose/upload an image file
    image_path = select_image_file()
    if not image_path or not os.path.exists(image_path):
        print(f"Error: Selected image file path does not exist or is empty: '{image_path}'")
        return

    # Define the arguments programmatically
    parser = utility.init_args()
    args = parser.parse_args([
        "--use_gpu=False",
        f"--image_dir={image_path}",
        "--rec_model_dir=./models/digit_only_rec",
        "--rec_char_dict_path=./dataset/digit_dict.txt",
        "--rec_image_shape=3,48,320"
    ])
    
    # Initialize the recognizer
    recognizer = TextRecognizer(args)
    
    # Load and check the image
    img, flag, _ = check_and_read(image_path)
    if not flag:
        img = cv2.imread(image_path)
        
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    # Perform recognition
    results, elapsed_time = recognizer([img])
    
    print("\n" + "=" * 50)
    print("OCR Prediction Result:")
    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Text predicted: {results[0][0]}")
    print(f"Confidence score: {results[0][1]:.4f}")
    print(f"Inference time: {elapsed_time:.4f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    main()
