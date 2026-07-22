# AMR (Automated Meter Reading) — Research Paper Structure & CVAA 2026 Submission Guide

> **Verify before you commit:** Confirm CVAA 2026's indexing status directly on IEEE's official conference search (https://ieeexplore.ieee.org/search/searchresult.jsp) before paying any submission/registration fee. Flyers claiming indexing are not the same as a confirmed IEEE Xplore conference record.

---

## PART A — Conference Requirements (from the flyer, as stated)

| Item | Detail |
|---|---|
| Conference | 6th International Conference on Computer Vision, Application and Algorithm (CVAA 2026) |
| Dates | September 18–20, 2026 |
| Location | Hong Kong, China |
| Website | www.iccvaa.org |
| Full Paper Deadline | September 4, 2026 |
| Final Paper Deadline | September 14, 2026 |
| Registration Deadline | September 11, 2026 |
| Relevant Track | Track 1: Computer Vision (primary) — could also cross-list under Track 2: Computer Algorithm given the CTC/NRTR decoding and dictionary-restriction contribution |
| Claimed Indexing | IEEE Xplore, EI Compendex, Scopus (**verify independently**) |
| Contact | Ms. Li — WhatsApp/Tel/WeChat +86-17702011266 or +8618981286513, iccvaa@iaeee.com |
| Review Process | Stated as 2–3 rounds of double-blind peer review |

**Action items before submitting:**
1. Download the *official* IEEE conference paper template (not a generic one) directly from www.iccvaa.org's submission page.
2. Check the submission system name/link on the site — legitimate IEEE-sponsored conferences typically use EDAS, Microsoft CMT, or EasyChair. Confirm which one CVAA 2026 uses rather than emailing a paper directly.
3. Ask organizers (in writing, not WhatsApp only) for the exact IEEE Conference Record Number — this is checkable in IEEE Xplore's database.
4. Check that Track 1 or Track 3 genuinely covers OCR/edge-AI meter reading (yours fits well — object detection + recognition + edge deployment).

---

## PART B — Standard IEEE Conference Paper Formatting Rules

These are the near-universal formatting conventions for IEEE conference proceedings (verify final numbers against CVAA 2026's own template once downloaded, since some details vary conference-to-conference):

### Page Layout
- **Paper size:** US Letter (8.5" × 11")
- **Columns:** Two-column format
- **Margins:** Top 0.75", Bottom 1", Side 0.625" (typical IEEE default — confirm with their template)
- **Page limit:** Typically 6 pages including references, figures, and tables. Extra pages often allowed for a fee (commonly 2 extra pages max) — confirm with organizers.

### Fonts
- **Body text:** Times New Roman, 10pt
- **Title:** 24pt, centered, Title Case
- **Author names:** 11pt, centered
- **Affiliation/email:** 9pt, centered, italics for affiliation
- **Section headings:** 10pt, ALL CAPS, bold, centered (I., II., III. numbering)
- **Subsection headings:** 10pt, italic, left-aligned (A., B., C. numbering)
- **Abstract & Index Terms:** 9pt, bold "Abstract—" and "Index Terms—" lead-ins

### Figures & Tables
- Captions **below** figures, **above** tables
- Caption font: 8pt
- All figures/tables must be referenced in text before they appear
- Numbered sequentially (Fig. 1, Fig. 2 / Table I, Table II — tables use Roman numerals by IEEE convention)
- Resolution: minimum 300 DPI for print-quality raster images; vector (EPS/PDF) preferred for diagrams

### Equations
- Centered, numbered sequentially in parentheses flush right: `(1)`, `(2)`, etc.
- Referenced in text as "Eq. (1)" or "(1)"

### Citations & References
- **In-text:** numbered bracketed style, e.g., `[1]`, `[2], [3]`
- **Reference list:** numbered in order of first citation (not alphabetical), IEEE style:
  ```
  [1] A. Author, B. Author, "Title of paper," in Proc. Conf. Name, City, Country, Year, pp. xx–xx.
  [2] A. Author, "Title of journal article," Journal Name, vol. X, no. Y, pp. xx–xx, Month Year.
  ```

### File Submission
- Camera-ready typically requires: source file (Word/LaTeX), PDF, IEEE copyright form (eCF), and PDF eXpress validation (IEEE PDF eXpress checks Xplore-compatibility — CVAA should provide a Conference ID for this).

---

## PART C — Full Paper Structure (Mapped to Your AMR Project)

### Title Page

**Title:**
*Domain-Specific Fine-Tuning of Lightweight Vision-Language OCR Models for Industrial Meter Reading Applications*

*(Optional alt titles if you want variety — pick one, don't submit multiple):*
- *Constrained-Vocabulary OCR for Real-Time Automated Meter Reading on Edge Devices*
- *YOLOv8-Guided Digit Recognition Pipeline for Mechanical Utility Meters: A Comparative Study of CNN and PP-OCRv4 Approaches*

**Authors & Affiliations:** [Your name(s)], [Department], [Institution], [City, Country], [email]

**Abstract (150–250 words, 9pt, no citations inside it):**
Use the draft you already have — trim to conference word limit. Must state: problem → gap in existing OCR → your method → key result numbers → significance.

**Index Terms (5–6 keywords):**
Automated Meter Reading, Optical Character Recognition, YOLOv8, PP-OCRv4, SVTR_LCNet, Edge AI Deployment

---

### I. Introduction
- Real-world problem: manual/error-prone utility meter reading, need for automation
- Why standard OCR fails on mechanical meter digits (font distortion, glare, rolling digits, no fixed vocabulary control)
- Gap: open-vocabulary OCR hallucinates non-numeric characters; segmentation-based CNNs are brittle to mechanical roll artifacts
- **Your contribution (state explicitly as a numbered list):**
  1. A YOLOv8-based ROI detection stage for isolating the 5-digit counter window
  2. A domain-constrained PP-OCRv4 (SVTR_LCNet) recognition pipeline with a locked 10-class digit dictionary
  3. A comparative evaluation against a segmentation + CNN baseline
  4. A validated real-time edge deployment (Android/ONNX Runtime, Jetson Nano/TensorRT) under 20ms inference

### II. Related Work
- Prior AMR/OCR literature (classical digit-segmentation approaches, CRNN/CTC-based scene text recognition, PaddleOCR family, YOLO-based ROI detection in industrial settings)
- Position your work: most existing OCR work targets open-vocabulary scene text; yours targets closed, fixed-length numeric domains — explain why that distinction matters for accuracy guarantees

### III. Proposed Methodology
**A. System Overview** — one diagram (your pipeline diagram, cleaned up as a proper Fig. 1)
**B. Stage 1: ROI Detection (YOLOv8)** — architecture, training data, bounding box output
**C. Stage 2: Preprocessing** — deskewing, OpenCV pipeline, zero-padding label formatting
**D. Stage 3a: Baseline — Segmentation + CNN (VGG)** — digit slicing method, CNN classifier design
**E. Stage 3b: Proposed — Domain-Constrained PP-OCRv4** — SVTR_LCNet backbone, PPLCNetV3 features, dual CTC+NRTR heads, 10-class dictionary constraint (`digit_dict.txt`), max_text_length=5
**F. Stage 4: Edge Deployment** — ONNX export via paddle2onnx, Android/Kotlin integration, Jetson Nano/TensorRT path

### IV. Experimental Setup
- Dataset description: source, size, train/valid/test split, annotation method
- Ground truth formatting (zero-padding example)
- Hardware/software: GPU used for training, ONNX Runtime version, Jetson Nano specs
- Evaluation metrics: full-string exact-match accuracy, per-digit accuracy, inference latency (ms), model size (MB)

### V. Results and Discussion
- **Table I:** Exact-match accuracy — CNN baseline vs. PP-OCRv4 approach
- **Table II:** Inference latency and model size comparison across platforms (desktop GPU, Android ONNX Runtime, Jetson Nano)
- **Fig. X:** Confusion matrix on visually ambiguous digit pairs (3 vs. 8, 0 vs. 6, 5 vs. 6)
- Discussion: why the constrained vocabulary eliminates a specific error class (non-numeric hallucination) — quantify how often the baseline produced non-numeric outputs if you have that data
- Failure case analysis: what still breaks the system (motion blur, extreme rotation, partial occlusion)

### VI. Conclusion and Future Work
- Summarize accuracy + latency achievement in one sentence with numbers
- Future work: temporal smoothing across video frames, active learning on hard examples, expanding to non-digit meter types (analog dial gauges)

### Acknowledgment
(Optional — funding, institution, advisor)

### References
IEEE numbered format, minimum 15–25 references for a conference paper of this scope. Must include: YOLOv8 original/documentation, PaddleOCR/PP-OCRv4 papers, SVTR paper, CTC loss paper (Graves et al.), at least 3–5 prior AMR-specific papers, ONNX/TensorRT deployment references.

---

## PART D — Pre-Submission Checklist

- [ ] Downloaded official CVAA 2026 IEEE template from www.iccvaa.org
- [ ] Verified conference legitimacy via IEEE Xplore conference search
- [ ] Paper within page limit (check exact number on their site — do not assume 6 pages)
- [ ] All figures ≥300 DPI, referenced in text before appearing
- [ ] All tables use Roman numeral labels (Table I, II...)
- [ ] References in IEEE numbered style, ordered by first citation
- [ ] Abstract within stated word limit, no citations inside it
- [ ] 5–6 index terms included
- [ ] Anonymized for double-blind review if required (remove author names/affiliations from body, self-citations phrased in third person)
- [ ] Plagiarism/similarity check run (many IEEE conferences use CrossCheck/iThenticate — similarity threshold typically <15–20%, confirm exact policy)
- [ ] Submission system account created (EDAS/CMT/EasyChair — whichever CVAA 2026 actually uses)
- [ ] Camera-ready: IEEE PDF eXpress validation done, copyright form (eCF) signed, at least one author registered by the registration deadline

---

*Note: Numeric formatting details (exact margins, page limits, template version) should be cross-checked against the actual .docx/.tex template you download from www.iccvaa.org — conference organizers sometimes deviate slightly from the IEEE default, and their template is the authoritative source, not this guide.*
