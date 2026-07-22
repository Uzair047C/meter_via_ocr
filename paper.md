# Project & Conference Reference

## Conference Details
- Conference: 6th CVAA 2026 (International Conference on Computer Vision, Application and Algorithm)
- Dates: September 18-20, 2026
- Location: Hong Kong, China
- Website: http://www.iccvaa.org
- Contact: Ms. Li (+8618981286513 / iccvaa@iaeee.com)
- Target Tracks: Track 1 (Computer Vision) & Track 3 (Computer Applications)

---

## Paper Submission Info

### Title
Domain-Specific Fine-Tuning of Lightweight Vision-Language OCR Models for Industrial Meter Reading Applications

### Abstract
Utility meter digit recognition presents unique challenges for standard Optical Character Recognition (OCR) systems due to severe visual degradation, uneven lighting, non-standard mechanical font profiles, and fixed-length numeric constraints. Standard open-vocabulary OCR pipelines often fail by hallucinating alphabetical or special characters into string predictions. In this paper, we propose a lightweight end-to-end digit-only recognition pipeline optimized for real-time edge deployment on automated meter reading (AMR) systems. Built upon the SVTR_LCNet architecture within the PP-OCRv4 framework, our approach enforces strict output vocabulary constraints directly at the model classification head via a custom 10-class character dictionary ($0\text{--}9$), eliminating out-of-vocabulary inference errors by design. We evaluate performance across custom dataset splits containing cropped 5-digit mechanical meter readings, comparing fine-tuned exact-match recognition accuracy against legacy convolutional neural network (CNN) architectures and traditional segmentation methods. Furthermore, we demonstrate model quantization and cross-platform optimization strategies using ONNX Runtime for Android and TensorRT/Paddle-Lite for edge hardware (such as NVIDIA Jetson Nano), achieving real-time inference latency under $20\text{ ms}$ per frame while maintaining high exact-match recognition fidelity on degraded real-world inputs.

---

## Proposed Paper Outline
1. Introduction: Automated Meter Reading (AMR) challenges, mechanical counter degradation, limitations of open-vocabulary OCR vs. constrained dictionary models.
2. Dataset & Preprocessing: Bounding-box cropping, zero-padded 5-digit ground truth formatting, image deskewing and pre-filtering.
3. Proposed Methodology: SVTR_LCNet backbone, custom 10-class dictionary head constraint, combined CTC + NRTR loss functions.
4. Edge Deployment & Quantization: ONNX Runtime export for Android and Paddle-Lite runtime configuration for edge AI devices (Jetson Nano).
5. Experiments & Results: Full 5-digit exact-match rate vs. individual digit accuracy, confusion matrix across visual digit pairs (e.g., 3 vs. 8, 0 vs. 6), and hardware latency benchmarks.

---

## Instructions for AI Assistant (To use tomorrow)
"Using the paper abstract, title, and outline provided above, please help me [draft section X / expand on methodology / write introduction / format citations]."

```markdown
# Everything About .md Files & Requirements

## What is an .md File?

An **.md file** is a **Markdown** file. Markdown is a lightweight plain-text formatting language used by developers, researchers, and technical writers worldwide.

Instead of heavy binary formats like `.docx` or `.pdf`, `.md` files contain clean text with simple symbols (like `#`, `**`, `-`) that instruct renderers how to style headings, bold text, bullet points, and code blocks.

### Key Syntax Elements

- **Headings**: `# Heading 1`, `## Heading 2`, `### Heading 3`
- **Bold / Italics**: `**bold text**`, `*italic text*`
- **Lists**: `- item 1` or `1. item 1`
- **Code Blocks**: Enclosed in triple backticks
  ```python
  # Example code
  print("Hello, World!")
  ```
- **LaTeX Math**: `$inline math$` or `$$display math equation$$`

## Requirements & Software to Open/Edit .md Files

You don't need expensive software. You can open and edit `.md` files on any system using:

1. **VS Code (Recommended)**: Install Visual Studio Code with the extensions **"Markdown All in One"** or **"Markdown Preview Enhanced"**. It gives you side-by-side live previews and LaTeX rendering.

2. **Dedicated Editors**: Obsidian, Typora, or Notion.

3. **GitHub / GitLab**: Any `.md` file uploaded to GitHub (like your `README.md`) renders directly as a webpage.

4. **Basic Text Editors**: Notepad, TextEdit, or Nano (though you won't see rendered formatting).

---

# Complete Architectural Overview of Your AMR Project  
**(YOLOv8 + CNN Segmentation Classifier + Fine-Tuned PaddleOCR)**

Your project is an **Automated Meter Reading (AMR) System** designed to take raw images of mechanical utility meters and convert them into clean **5-digit numerical readings**.

## System Architecture

```mermaid
flowchart TD
    A[Full Input Image<br>(Meter Photo)] --> B[STAGE 1: ROI Detection<br>(YOLOv8)]
    B --> C[Cropped 5-Digit Counter Strip]
    C --> D[STAGE 2: Preprocessing<br>(Deskew, OpenCV, Crop)]
    D --> E{Recognition Approach}
    E --> F[APPROACH A: CNNs<br>(Digit Segmentation)]
    E --> G[APPROACH B: PaddleOCR<br>(Fine-Tuned Recognition)]
    F --> H[5 Single-Digit Predictions]
    G --> H
    H --> I[STAGE 3: Model Export & Edge Deployment<br>(Android / Jetson)]
```

### Stage 1: Region of Interest (ROI) Detection (YOLOv8)

- **Goal**: Locate and crop the exact counter strip window from a full high-resolution photo of a physical meter.
- **Why YOLOv8?** YOLO operates as a real-time object detector. In a full meter image, there is noise (reflections, plastic housing, pipes). YOLOv8 draws a tight bounding box around just the **5-digit dial area**.
- **Output**: A single cropped sub-image containing only the digit strip (e.g., shape `[48 x 320]`).

### Stage 2: Image Preprocessing & Dataset Management

Before recognition, the cropped strip undergoes filtering:

1. **Deskewing & Alignment**: Correcting slight rotational angles from field photos.
2. **Annotation Tool**: Custom OpenCV + Python annotation script that coordinates bounding box coordinates and manages label files (`train.csv`, `valid.csv`, `test.csv`).
3. **Zero-Padding Formatting**: Standardizing meter readings so every ground truth label is strictly **5 characters** (e.g., `17` → `'00017'`).

### Stage 3: Digit Recognition (Two Parallel Approaches)

#### Approach A: Traditional CNN Digit Segmentation (VGG Backbone)

1. **Segmentation**: Slice the cropped strip horizontally into **5 individual bounding boxes** (1×5 grid).
2. **Classification**: Pass each single-digit image into a custom CNN / VGG classifier trained on single-digit images (0–9).
3. **Re-assembly**: Combine the 5 separate predicted class IDs into a single 5-digit string.

> **Trade-off**: Works well for well-spaced digits, but struggles if mechanical wheels are mid-rotation or touching each other.

#### Approach B: Domain-Constrained PaddleOCR (PP-OCRv4 fine-tuning)

1. **End-to-End Recognition**: Reads the entire cropped strip directly without explicit individual character segmentation.
2. **Architecture**: SVTR_LCNet backbone with PPLCNetV3 feature extraction and dual heads (CTCHead + NRTRHead).
3. **Custom Dictionary Constraint** (`digit_dict.txt`): The output vocabulary is strictly locked to **10 classes** (0, 1, 2, 3, 4, 5, 6, 7, 8, 9). This prevents the model from ever predicting letters like 'O', 'l', or special characters.
4. **Max Length Cap**: Fixed at `max_text_length: 5`.

### Stage 4: Cross-Platform Edge Deployment (Android & Jetson)

To run this system on physical mobile/edge hardware in real time:

1. **ONNX Conversion**: Convert Paddle inference models (`.pdmodel` / `.pdiparams`) into open `.onnx` weights via `paddle2onnx`.
2. **Android App (Kotlin / ONNX Runtime)**:
   - Load `digit_rec.onnx` and `digit_dict.txt` inside Android assets.
   - Normalize camera frame bitmap into float buffer tensor `[1, 3, 48, 320]`.
   - Perform CTC greedy decoding in native Kotlin code to yield the 5-digit string in under **20 ms**.
3. **NVIDIA Jetson Nano**: Run optimized inference via **TensorRT** / **Paddle-Lite** for continuous video stream monitoring.

---

**Document generated for the AMR (Automated Meter Reading) project documentation.**
```

You can copy the content above and save it as `AMR_Project_Documentation.md`. It is fully formatted with proper headings, lists, code blocks, and a Mermaid diagram for the architecture. Open it in VS Code, Obsidian, or GitHub for the best rendered view.
```markdown
# Everything About .md Files & Requirements

## What is an .md File?

An **.md file** is a **Markdown** file. Markdown is a lightweight plain-text formatting language used by developers, researchers, and technical writers worldwide.

Instead of heavy binary formats like `.docx` or `.pdf`, `.md` files contain clean text with simple symbols (like `#`, `**`, `-`) that instruct renderers how to style headings, bold text, bullet points, and code blocks.

### Key Syntax Elements

- **Headings**: `# Heading 1`, `## Heading 2`, `### Heading 3`
- **Bold / Italics**: `**bold text**`, `*italic text*`
- **Lists**: `- item 1` or `1. item 1`
- **Code Blocks**: Enclosed in triple backticks
  ```python
  # Example code
  print("Hello, World!")
  ```
- **LaTeX Math**: `$inline math$` or `$$display math equation$$`

## Requirements & Software to Open/Edit .md Files

You don't need expensive software. You can open and edit `.md` files on any system using:

1. **VS Code (Recommended)**: Install Visual Studio Code with the extensions **"Markdown All in One"** or **"Markdown Preview Enhanced"**. It gives you side-by-side live previews and LaTeX rendering.

2. **Dedicated Editors**: Obsidian, Typora, or Notion.

3. **GitHub / GitLab**: Any `.md` file uploaded to GitHub (like your `README.md`) renders directly as a webpage.

4. **Basic Text Editors**: Notepad, TextEdit, or Nano (though you won't see rendered formatting).

---

# Complete Architectural Overview of Your AMR Project  
**(YOLOv8 + CNN Segmentation Classifier + Fine-Tuned PaddleOCR)**

Your project is an **Automated Meter Reading (AMR) System** designed to take raw images of mechanical utility meters and convert them into clean **5-digit numerical readings**.

## System Architecture

```mermaid
flowchart TD
    A[Full Input Image<br>(Meter Photo)] --> B[STAGE 1: ROI Detection<br>(YOLOv8)]
    B --> C[Cropped 5-Digit Counter Strip]
    C --> D[STAGE 2: Preprocessing<br>(Deskew, OpenCV, Crop)]
    D --> E{Recognition Approach}
    E --> F[APPROACH A: CNNs<br>(Digit Segmentation)]
    E --> G[APPROACH B: PaddleOCR<br>(Fine-Tuned Recognition)]
    F --> H[5 Single-Digit Predictions]
    G --> H
    H --> I[STAGE 3: Model Export & Edge Deployment<br>(Android / Jetson)]
```

### Stage 1: Region of Interest (ROI) Detection (YOLOv8)

- **Goal**: Locate and crop the exact counter strip window from a full high-resolution photo of a physical meter.
- **Why YOLOv8?** YOLO operates as a real-time object detector. In a full meter image, there is noise (reflections, plastic housing, pipes). YOLOv8 draws a tight bounding box around just the **5-digit dial area**.
- **Output**: A single cropped sub-image containing only the digit strip (e.g., shape `[48 x 320]`).

### Stage 2: Image Preprocessing & Dataset Management

Before recognition, the cropped strip undergoes filtering:

1. **Deskewing & Alignment**: Correcting slight rotational angles from field photos.
2. **Annotation Tool**: Custom OpenCV + Python annotation script that coordinates bounding box coordinates and manages label files (`train.csv`, `valid.csv`, `test.csv`).
3. **Zero-Padding Formatting**: Standardizing meter readings so every ground truth label is strictly **5 characters** (e.g., `17` → `'00017'`).

### Stage 3: Digit Recognition (Two Parallel Approaches)

#### Approach A: Traditional CNN Digit Segmentation (VGG Backbone)

1. **Segmentation**: Slice the cropped strip horizontally into **5 individual bounding boxes** (1×5 grid).
2. **Classification**: Pass each single-digit image into a custom CNN / VGG classifier trained on single-digit images (0–9).
3. **Re-assembly**: Combine the 5 separate predicted class IDs into a single 5-digit string.

> **Trade-off**: Works well for well-spaced digits, but struggles if mechanical wheels are mid-rotation or touching each other.

#### Approach B: Domain-Constrained PaddleOCR (PP-OCRv4 fine-tuning)

1. **End-to-End Recognition**: Reads the entire cropped strip directly without explicit individual character segmentation.
2. **Architecture**: SVTR_LCNet backbone with PPLCNetV3 feature extraction and dual heads (CTCHead + NRTRHead).
3. **Custom Dictionary Constraint** (`digit_dict.txt`): The output vocabulary is strictly locked to **10 classes** (0, 1, 2, 3, 4, 5, 6, 7, 8, 9). This prevents the model from ever predicting letters like 'O', 'l', or special characters.
4. **Max Length Cap**: Fixed at `max_text_length: 5`.

### Stage 4: Cross-Platform Edge Deployment (Android & Jetson)

To run this system on physical mobile/edge hardware in real time:

1. **ONNX Conversion**: Convert Paddle inference models (`.pdmodel` / `.pdiparams`) into open `.onnx` weights via `paddle2onnx`.
2. **Android App (Kotlin / ONNX Runtime)**:
   - Load `digit_rec.onnx` and `digit_dict.txt` inside Android assets.
   - Normalize camera frame bitmap into float buffer tensor `[1, 3, 48, 320]`.
   - Perform CTC greedy decoding in native Kotlin code to yield the 5-digit string in under **20 ms**.
3. **NVIDIA Jetson Nano**: Run optimized inference via **TensorRT** / **Paddle-Lite** for continuous video stream monitoring.

---

**Document generated for the AMR (Automated Meter Reading) project documentation.**
```

You can copy the content above and save it as `AMR_Project_Documentation.md`. It is fully formatted with proper headings, lists, code blocks, and a Mermaid diagram for the architecture. Open it in VS Code, Obsidian, or GitHub for the best rendered view.