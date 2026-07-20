import os
import sys
import time
import cv2
import numpy as np
from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename

# -------------------------------------------------------------------------
# 1. PATH CONFIGURATION & PADDLEOCR ENGINE SEEDING
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(current_dir, "tools")
sys.path.insert(0, tools_dir)
sys.path.insert(0, current_dir)

import utility1 as utility
from predict_rec import TextRecognizer
from ppocr.utils.utility import check_and_read

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the Recognizer globally on app startup for snappy interface performance
def init_ocr_engine():
    parser = utility.init_args()
    args = parser.parse_args([
        "--use_gpu=False",
        f"--image_dir={current_dir}",
        "--rec_model_dir=./models/digit_only_rec",
        "--rec_char_dict_path=./dataset/digit_dict.txt",
        "--rec_image_shape=3,48,320"
    ])
    return TextRecognizer(args)

print("Initializing Neural Network Engine...")
recognizer = init_ocr_engine()
print("Engine successfully armed.")

# Helper function to load validation data labels
def load_labels(label_file_path):
    labels = {}
    if not os.path.exists(label_file_path):
        return labels
    with open(label_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) >= 2:
                img_name = os.path.basename(parts[0].strip())
                labels[img_name] = parts[1].strip()
    return labels

# -------------------------------------------------------------------------
# 2. UI/UX DESIGN JINJA TEMPLATE (HTML, CSS, JS)
# -------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PaddleOCR Digit Intel Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #FAF7F2;
            --bg-surface: #FFFFFF;
            --primary: #FF6F00;
            --primary-hover: #E65F00;
            --accent-orange: #FF8F00;
            --text-main: #3E2723;
            --text-muted: #795548;
            --border-color: #EFEBE9;
            --brown-soft: #D2B48C;
            --shadow-sm: 0 2px 8px rgba(62, 39, 35, 0.04);
            --shadow-md: 0 10px 30px rgba(62, 39, 35, 0.08);
            --radius-lg: 16px;
            --radius-md: 10px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background-color: var(--bg-base); color: var(--text-main); min-height: 100vh; padding: 2rem 1rem; }
        .container { max-width: 1000px; margin: 0 auto; }
        
        /* Brand Header */
        header { text-align: center; margin-bottom: 2.5rem; }
        header h1 { font-size: 2.5rem; font-weight: 700; color: var(--text-main); letter-spacing: -0.5px; }
        header h1 span { color: var(--primary); }
        header p { color: var(--text-muted); margin-top: 0.5rem; font-weight: 400; }

        /* UI Tabs Navigation */
        .tabs { display: flex; gap: 0.5rem; margin-bottom: 2rem; background: #EEE7DE; padding: 0.4rem; border-radius: var(--radius-md); max-width: 450px; margin-left: auto; margin-right: auto; }
        .tab-btn { flex: 1; border: none; padding: 0.75rem 1rem; background: transparent; cursor: pointer; font-weight: 600; font-size: 0.95rem; color: var(--text-muted); border-radius: 8px; transition: all 0.2s ease; }
        .tab-btn.active { background: var(--bg-surface); color: var(--primary); box-shadow: var(--shadow-sm); }

        /* Content Card Panels */
        .card { background: var(--bg-surface); border-radius: var(--radius-lg); padding: 2.5rem; box-shadow: var(--shadow-md); border: 1px solid var(--border-color); display: none; }
        .card.active { display: block; animation: fadeIn 0.4s ease; }

        /* Dropzones & Controls */
        .dropzone { border: 2px dashed var(--brown-soft); border-radius: var(--radius-md); padding: 3rem 2rem; text-align: center; background: #FCFAF7; cursor: pointer; transition: all 0.2s ease; position: relative; }
        .dropzone:hover { border-color: var(--primary); background: #FFF9F5; }
        .dropzone svg { width: 48px; height: 48px; fill: var(--accent-orange); margin-bottom: 1rem; }
        .dropzone input { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
        
        .form-group { margin-bottom: 1.5rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: 600; font-size: 0.9rem; color: var(--text-main); }
        .form-control { width: 100%; padding: 0.85rem 1rem; border: 1px solid var(--brown-soft); border-radius: var(--radius-md); background: #FAF7F2; color: var(--text-main); font-size: 0.95rem; transition: border-color 0.2s; }
        .form-control:focus { outline: none; border-color: var(--primary); background: #FFF; }

        .btn { display: inline-flex; align-items: center; justify-content: center; width: 100%; background: var(--primary); color: #fff; border: none; padding: 1rem; font-size: 1rem; font-weight: 600; border-radius: var(--radius-md); cursor: pointer; transition: background 0.2s ease, transform 0.1s; margin-top: 1rem; }
        .btn:hover { background: var(--primary-hover); }
        .btn:active { transform: scale(0.99); }

        /* Modern Metrics Display GRID */
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 2rem; }
        .metric-card { background: #FCFAF7; padding: 1.5rem; border-radius: var(--radius-md); border: 1px solid var(--border-color); text-align: center; position: relative; overflow: hidden; }
        .metric-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--accent-orange); }
        .metric-label { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: var(--text-main); margin-top: 0.5rem; }
        .metric-value.highlight { color: var(--primary); }

        /* Loader */
        .loader { display: none; margin: 2rem auto 0; width: 40px; height: 40px; border: 4px solid var(--border-color); border-top: 4px solid var(--primary); border-radius: 50%; animation: spin 1s linear infinite; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Test Folder Table styling */
        .results-table-container { max-height: 500px; overflow-y: auto; margin-top: 1.5rem; border-radius: var(--radius-md); border: 1px solid var(--border-color); }
        .results-table { width: 100%; border-collapse: collapse; background: var(--bg-surface); text-align: left; }
        .results-table th, .results-table td { padding: 1rem; border-bottom: 1px solid var(--border-color); vertical-align: middle; }
        .results-table th { background: #EEE7DE; font-weight: 700; color: var(--text-main); position: sticky; top: 0; z-index: 10; }
        .results-table tr:hover { background: #FCFAF7; }
        
        .status-badge { display: inline-flex; align-items: center; padding: 0.35rem 0.75rem; border-radius: 20px; font-weight: 700; font-size: 0.85rem; }
        .status-badge.correct { background: #E8F5E9; color: #2E7D32; border: 1px solid #C8E6C9; }
        .status-badge.wrong { background: #FFEBEE; color: #C62828; border: 1px solid #FFCDD2; }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>PaddleOCR <span>Intelligence Panel</span></h1>
        <p>A unified interface for deep-learning single digit recognition and batch fold evaluation</p>
    </header>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('single')">Single Prediction</button>
        <button class="tab-btn" onclick="switchTab('batch')">Folder Evaluation</button>
        <button class="tab-btn" onclick="switchTab('test-folder')">Test Folder View</button>
    </div>

    <!-- SINGLE RECOGNITION INTERFACE -->
    <div id="single-panel" class="card active">
        <form id="single-form">
            <div class="dropzone">
                <svg viewBox="0 0 24 24"><path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/></svg>
                <p style="font-weight:600; margin-bottom:4px;">Drag and drop system image here</p>
                <p style="font-size:0.85rem; color:var(--text-muted);">Supports JPG, PNG, BMP, TIFF</p>
                <input type="file" name="file" id="file-input" onchange="updateFileName(this)">
                <div id="file-name-display" style="margin-top:10px; font-weight:600; color:var(--primary);"></div>
            </div>
            <button type="submit" class="btn">Execute Prediction</button>
        </form>

        <div id="single-loader" class="loader"></div>

        <div id="single-results" style="display:none;">
            <div style="text-align: center; margin-top: 2rem;">
                <p style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;">Selected Image</p>
                <img id="res-img" src="" alt="Selected Image" style="max-width: 100%; max-height: 220px; border-radius: var(--radius-md); border: 1px solid var(--brown-soft); box-shadow: var(--shadow-sm); background-color: #fafafa; padding: 4px;" />
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Predicted Output</div>
                    <div class="metric-value highlight" id="res-text">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-value" id="res-conf">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Latency Time</div>
                    <div class="metric-value" id="res-time">-</div>
                </div>
            </div>
        </div>
    </div>

    <!-- BATCH EVALUATION INTERFACE -->
    <div id="batch-panel" class="card">
        <form id="batch-form">
            <div class="form-group">
                <label>System Evaluation Fold Target Directory Path</label>
                <input type="text" name="image_dir" class="form-control" placeholder="E.g., D:/PaddleOCR/ocr_app/dataset/images/test">
            </div>
            <div class="form-group">
                <label>Validation Ground Truth Mapping File Filepath (.txt)</label>
                <input type="text" name="label_file" class="form-control" placeholder="E.g., D:/PaddleOCR/ocr_app/dataset/labels/test_list.txt">
            </div>
            <button type="submit" class="btn">Compute Dynamic Metrics</button>
        </form>

        <div id="batch-loader" class="loader"></div>

        <div id="batch-results" style="display:none;">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Exact Match Accuracy</div>
                    <div class="metric-value highlight" id="batch-exact">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Char-Level Accuracy</div>
                    <div class="metric-value highlight" id="batch-char">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Core Confidence</div>
                    <div class="metric-value" id="batch-conf">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Aggregated System Capacity</div>
                    <div class="metric-value" id="batch-total">-</div>
                </div>
            </div>
        </div>
    </div>

    <!-- TEST FOLDER VIEW INTERFACE -->
    <div id="test-folder-panel" class="card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 10px;">
            <h2 style="font-size: 1.3rem; font-weight: 700; color: var(--text-main);">Test Folder Evaluation Details</h2>
            <div id="test-folder-stats" style="font-weight: 600; color: var(--text-muted); font-size: 0.95rem;">
                Processed: <span id="test-loaded-count" style="color: var(--primary);">0</span>/208 | Accuracy: <span id="test-acc-val" style="color: var(--primary);">-</span>
            </div>
        </div>
        
        <div id="test-folder-loader" class="loader"></div>
        
        <div id="test-table-wrapper" class="results-table-container" style="display:none;">
            <table class="results-table">
                <thead>
                    <tr>
                        <th style="width: 25%;">Image</th>
                        <th style="width: 25%;">Actual Label</th>
                        <th style="width: 25%;">Predicted Label</th>
                        <th style="width: 25%;">Status</th>
                    </tr>
                </thead>
                <tbody id="test-table-body">
                    <!-- Dynamic Rows -->
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
    function switchTab(mode) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
        if(mode === 'single') {
            document.querySelectorAll('.tab-btn')[0].classList.add('active');
            document.getElementById('single-panel').classList.add('active');
        } else if(mode === 'batch') {
            document.querySelectorAll('.tab-btn')[1].classList.add('active');
            document.getElementById('batch-panel').classList.add('active');
        } else if(mode === 'test-folder') {
            document.querySelectorAll('.tab-btn')[2].classList.add('active');
            document.getElementById('test-folder-panel').classList.add('active');
            loadTestFolderResults();
        }
    }

    let testFolderLoaded = false;
    async function loadTestFolderResults() {
        if (testFolderLoaded) return;
        
        const loader = document.getElementById('test-folder-loader');
        const wrapper = document.getElementById('test-table-wrapper');
        const tbody = document.getElementById('test-table-body');
        
        loader.style.display = 'block';
        wrapper.style.display = 'none';
        
        try {
            const res = await fetch('/get_test_results');
            const data = await res.json();
            if(data.error) throw new Error(data.error);
            
            tbody.innerHTML = '';
            let correctCount = 0;
            
            data.forEach(item => {
                const tr = document.createElement('tr');
                
                // 1st column: Image
                const tdImg = document.createElement('td');
                const img = document.createElement('img');
                img.src = `/test_images/${item.filename}`;
                img.alt = item.filename;
                img.style.maxHeight = '40px';
                img.style.borderRadius = '4px';
                img.style.border = '1px solid var(--border-color)';
                tdImg.appendChild(img);
                
                // 2nd column: Actual Label
                const tdActual = document.createElement('td');
                tdActual.innerText = item.actual;
                tdActual.style.fontWeight = '600';
                
                // 3rd column: Predicted Label
                const tdPredicted = document.createElement('td');
                tdPredicted.innerText = item.predicted;
                tdPredicted.style.fontWeight = '700';
                tdPredicted.style.color = 'var(--primary)';
                
                // 4th column: Correct or Wrong
                const tdStatus = document.createElement('td');
                const badge = document.createElement('span');
                badge.className = `status-badge ${item.status.toLowerCase()}`;
                
                if (item.status === 'Correct') {
                    correctCount++;
                    badge.innerHTML = `
                        <svg viewBox="0 0 24 24" style="width:16px; height:16px; fill:#2E7D32; margin-right:4px; display:inline-block; vertical-align:middle;">
                            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                        </svg>
                        Correct
                    `;
                } else {
                    badge.innerHTML = `
                        <svg viewBox="0 0 24 24" style="width:16px; height:16px; fill:#C62828; margin-right:4px; display:inline-block; vertical-align:middle;">
                            <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                        </svg>
                        Wrong
                    `;
                }
                tdStatus.appendChild(badge);
                
                tr.appendChild(tdImg);
                tr.appendChild(tdActual);
                tr.appendChild(tdPredicted);
                tr.appendChild(tdStatus);
                tbody.appendChild(tr);
            });
            
            document.getElementById('test-loaded-count').innerText = data.length;
            const accuracy = ((correctCount / data.length) * 100).toFixed(2) + "%";
            document.getElementById('test-acc-val').innerText = accuracy;
            
            testFolderLoaded = true;
            wrapper.style.display = 'block';
        } catch(err) {
            alert(err.message || "Failed to load test folder results.");
        } finally {
            loader.style.display = 'none';
        }
    }

    function updateFileName(input) {
        const display = document.getElementById('file-name-display');
        display.innerText = input.files.length > 0 ? "Targeted File: " + input.files[0].name : "";
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                document.getElementById('res-img').src = e.target.result;
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    // Process Single Form Data Submission
    document.getElementById('single-form').onsubmit = async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('file-input');
        if(fileInput.files.length === 0) return alert("Please map an image first.");

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        document.getElementById('single-loader').style.display = 'block';
        document.getElementById('single-results').style.display = 'none';

        try {
            const res = await fetch('/predict_single', { method: 'POST', body: formData });
            const data = await res.json();
            if(data.error) throw new Error(data.error);

            document.getElementById('res-text').innerText = data.text;
            document.getElementById('res-conf').innerText = (data.confidence * 100).toFixed(2) + "%";
            document.getElementById('res-time').innerText = data.elapsed_time.toFixed(4) + "s";
            document.getElementById('single-results').style.display = 'block';
        } catch(err) {
            alert(err.message || "Execution Pipeline Crash.");
        } finally {
            document.getElementById('single-loader').style.display = 'none';
        }
    };

    // Process Evaluation Batch Run
    document.getElementById('batch-form').onsubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);

        document.getElementById('batch-loader').style.display = 'block';
        document.getElementById('batch-results').style.display = 'none';

        try {
            const res = await fetch('/evaluate_batch', { method: 'POST', body: formData });
            const data = await res.json();
            if(data.error) throw new Error(data.error);

            document.getElementById('batch-exact').innerText = data.exact_match_acc.toFixed(2) + "%";
            document.getElementById('batch-char').innerText = data.char_level_acc.toFixed(2) + "%";
            document.getElementById('batch-conf').innerText = data.avg_confidence.toFixed(2) + "%";
            document.getElementById('batch-total').innerText = data.total_evaluated + " items";
            document.getElementById('batch-results').style.display = 'block';
        } catch(err) {
            alert(err.message || "Batch System Analysis Interrupted.");
        } finally {
            document.getElementById('batch-loader').style.display = 'none';
        }
    };
</script>
</body>
</html>
"""

# -------------------------------------------------------------------------
# 3. ROUTING BACKEND SYSTEM CONTROLLERS
# -------------------------------------------------------------------------
@app.route('/')
def dashboard():
    return render_template_string(HTML_TEMPLATE)

from flask import send_from_directory

test_results_cache = None

@app.route('/test_images/<path:filename>')
def serve_test_image(filename):
    test_dir = os.path.join(current_dir, "dataset", "images", "test")
    return send_from_directory(test_dir, filename)

@app.route('/get_test_results')
def get_test_results():
    global test_results_cache
    if test_results_cache is not None:
        return jsonify(test_results_cache)
        
    image_dir = os.path.join(current_dir, "dataset", "images", "test")
    label_file = os.path.join(current_dir, "dataset", "labels", "test_list.txt")
    
    labels = load_labels(label_file)
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_formats
    ])
    
    results_list = []
    
    # Process in batches
    batch_size = 16
    for i in range(0, len(image_files), batch_size):
        batch_names = image_files[i:i+batch_size]
        img_list = []
        valid_names = []
        
        for name in batch_names:
            fpath = os.path.join(image_dir, name)
            img, flag, _ = check_and_read(fpath)
            if not flag:
                img = cv2.imread(fpath)
            if img is not None:
                img_list.append(img)
                valid_names.append(name)
                
        if not img_list: continue
        batch_results, _ = recognizer(img_list)
        
        for name, (pred_text, confidence) in zip(valid_names, batch_results):
            gt_label = labels.get(name, "-")
            status = "Correct" if pred_text == gt_label else "Wrong"
            results_list.append({
                'filename': name,
                'actual': gt_label,
                'predicted': pred_text,
                'status': status
            })
            
    test_results_cache = results_list
    return jsonify(results_list)

@app.route('/predict_single', methods=['POST'])
def predict_single():
    if 'file' not in request.files:
        return jsonify({'error': 'No file segment found'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No active file designated'}), 400
    
    filename = secure_filename(file.filename)
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(fpath)

    try:
        # Load image via your specific utility verification logic
        img, flag, _ = check_and_read(fpath)
        if not flag:
            img = cv2.imread(fpath)
            
        if img is None:
            return jsonify({'error': 'Matrix breakdown mapping image file'}), 400

        # Perform recognition tracking execution timeline
        results, elapsed_time = recognizer([img])
        pred_text, confidence = results[0][0], float(results[0][1])

        # Safely clean up file storage metrics
        os.remove(fpath)

        return jsonify({
            'text': pred_text,
            'confidence': confidence,
            'elapsed_time': elapsed_time
        })
    except Exception as e:
        if os.path.exists(fpath): os.remove(fpath)
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate_batch', methods=['POST'])
def evaluate_batch():
    image_dir = request.form.get('image_dir', '').strip()
    label_file = request.form.get('label_file', '').strip()

    # Apply defaults if inputs are empty strings
    if not image_dir:
        image_dir = os.path.join(current_dir, "dataset", "images", "test")
    if not label_file:
        label_file = os.path.join(current_dir, "dataset", "labels", "test_list.txt")

    if not os.path.exists(image_dir):
        return jsonify({'error': f"Target processing directory paths '{image_dir}' do not exist."}), 400

    labels = load_labels(label_file)
    valid_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_formats
    ]

    if not image_files:
        return jsonify({'error': f'No valid image arrays verified inside target path {image_dir}'}), 400

    correct_count = 0
    total_evaluated = 0
    total_chars_correct = 0
    total_chars_possible = 0
    total_confidence = 0.0

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

        if not img_list: continue
        batch_results, _ = recognizer(img_list)

        for fpath, (pred_text, confidence) in zip(valid_batch_files, batch_results):
            filename = os.path.basename(fpath)
            gt_label = labels.get(filename, None)

            total_evaluated += 1
            total_confidence += confidence

            if gt_label is not None:
                is_correct = (pred_text == gt_label)
                if is_correct:
                    correct_count += 1
                
                matches = sum(1 for c1, c2 in zip(pred_text, gt_label) if c1 == c2)
                total_chars_correct += matches
                total_chars_possible += max(len(pred_text), len(gt_label))

    exact_match_acc = (correct_count / total_evaluated) * 100 if total_evaluated > 0 else 0
    char_level_acc = (total_chars_correct / total_chars_possible) * 100 if total_chars_possible > 0 else 0
    avg_confidence = (total_confidence / total_evaluated) * 100 if total_evaluated > 0 else 0

    return jsonify({
        'total_evaluated': total_evaluated,
        'exact_match_acc': exact_match_acc,
        'char_level_acc': char_level_acc,
        'avg_confidence': avg_confidence
    })

if __name__ == '__main__':
    # Start the integrated UI local runtime platform server
    app.run(host='127.0.0.1', port=5000, debug=True)