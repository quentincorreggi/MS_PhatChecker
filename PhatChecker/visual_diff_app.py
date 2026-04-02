#!/usr/bin/env python3
"""
Visual Regression Diff — Web UI
Run with: python3 visual_diff_app.py
Then open http://localhost:5005
"""

import os
import sys
import json
import threading
import time
import webbrowser
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, send_file

# Import the diff engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_diff import (
    run_batch, run_comparison, generate_html_report, load_mask_config
)

app = Flask(__name__)

# State
current_job = {"status": "idle", "progress": 0, "total": 0, "results": None, "message": ""}

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/api/browse", methods=["POST"])
def browse_dir():
    """List directory contents for the folder browser."""
    data = request.json
    path = data.get("path", os.path.expanduser("~"))
    path = os.path.expanduser(path)

    if not os.path.isdir(path):
        return jsonify({"error": f"Not a directory: {path}"}), 400

    items = []
    try:
        for entry in sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.name.startswith("."):
                continue
            items.append({
                "name": entry.name,
                "path": entry.path,
                "isDir": entry.is_dir(),
                "isPng": entry.name.lower().endswith(".png"),
            })
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    parent = str(Path(path).parent)
    return jsonify({"path": path, "parent": parent, "items": items})


@app.route("/api/sample-image", methods=["POST"])
def sample_image():
    """Return the first PNG from a directory for the mask editor."""
    data = request.json
    path = data.get("path", "")
    if not os.path.isdir(path):
        return jsonify({"error": "Not a directory"}), 400

    for f in sorted(os.listdir(path)):
        if f.lower().endswith(".png"):
            return send_file(os.path.join(path, f), mimetype="image/png")

    return jsonify({"error": "No PNG files found"}), 404


@app.route("/api/run", methods=["POST"])
def run_diff():
    """Start a batch comparison."""
    global current_job
    data = request.json

    old_dir = data.get("oldDir", "")
    new_dir = data.get("newDir", "")
    threshold = float(data.get("threshold", 0.995))
    workers = int(data.get("workers", 4))
    save_pass = data.get("savePass", False)
    color_normalize_k = int(data.get("colorNormalizeK", 0))
    masks = data.get("masks", [])
    shifts = data.get("shifts", [])
    mask_image_size = data.get("maskImageSize", None)
    output_dir = data.get("outputDir", "") or os.path.join(old_dir, "..", "regression_report")
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(old_dir):
        return jsonify({"error": f"Old directory not found: {old_dir}"}), 400
    if not os.path.isdir(new_dir):
        return jsonify({"error": f"New directory not found: {new_dir}"}), 400

    current_job = {"status": "running", "progress": 0, "total": 0, "results": None, "message": "Starting..."}

    def worker():
        global current_job
        try:
            mask_list = masks if masks else None
            shift_list = shifts if shifts else None
            msize = tuple(mask_image_size) if mask_image_size else None

            results, only_old, only_new = run_batch(
                old_dir, new_dir, output_dir, threshold, workers, save_pass,
                mask_list, msize, shift_list, color_normalize_k
            )

            report_path = generate_html_report(results, only_old, only_new, output_dir, threshold)

            failed = [r for r in results if not r["passed"]]
            swapped_list = [r for r in results if r["passed"] and r.get("color_swapped")]
            passed_list = [r for r in results if r["passed"] and not r.get("color_swapped")]

            swap_msg = f", {len(swapped_list)} color swaps" if swapped_list else ""
            current_job = {
                "status": "done",
                "progress": len(results),
                "total": len(results),
                "message": f"Done! {len(passed_list)} passed{swap_msg}, {len(failed)} failed",
                "results": {
                    "total": len(results),
                    "passed": len(passed_list),
                    "swapped": len(swapped_list),
                    "failed": len(failed),
                    "missing_old": len(only_old),
                    "missing_new": len(only_new),
                    "report_path": report_path,
                    "output_dir": output_dir,
                    "levels": [
                        {"level": r["level"], "score": float(r["score"]),
                         "passed": bool(r["passed"]),
                         "color_swapped": bool(r.get("color_swapped", False)),
                         "change_pct": float(r["change_pct"])}
                        for r in results
                    ],
                },
            }
        except Exception as e:
            current_job = {"status": "error", "message": str(e), "progress": 0, "total": 0, "results": None}

    t = threading.Thread(target=worker)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def job_status():
    return jsonify(current_job)


@app.route("/api/save-mask", methods=["POST"])
def save_mask():
    """Save mask config to a JSON file."""
    data = request.json
    save_path = data.get("path", "mask_config.json")
    config = {
        "masks": data.get("masks", []),
        "shifts": data.get("shifts", []),
        "imageWidth": data.get("imageWidth", 0),
        "imageHeight": data.get("imageHeight", 0),
    }
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    return jsonify({"saved": save_path})


@app.route("/report/<path:filepath>")
def serve_report(filepath):
    """Serve generated report files."""
    return send_from_directory("/", filepath)


# ---------------------------------------------------------------------------
# Full HTML UI
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Regression Diff Tool</title>
<style>
:root {
    --bg: #0f0f1a;
    --surface: #1a1a2e;
    --surface2: #16213e;
    --border: #2a2a4a;
    --text: #e0e0e0;
    --text2: #888;
    --accent: #6c63ff;
    --accent2: #4fc3f7;
    --green: #4caf50;
    --red: #f44336;
    --orange: #ff9800;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

/* Layout */
.app { display: flex; flex-direction: column; min-height: 100vh; }
.app-header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; gap: 16px; }
.app-header h1 { font-size: 20px; font-weight: 700; }
.app-header .subtitle { color: var(--text2); font-size: 13px; }

.tabs { display: flex; gap: 4px; background: var(--bg); padding: 4px; border-radius: 10px; }
.tab { padding: 8px 20px; border-radius: 8px; cursor: pointer; font-size: 14px; color: var(--text2); border: none; background: none; transition: all 0.15s; }
.tab:hover { color: var(--text); }
.tab.active { background: var(--accent); color: white; }

.main { flex: 1; padding: 24px; max-width: 1200px; margin: 0 auto; width: 100%; }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Forms */
.form-group { margin-bottom: 20px; }
.form-group label { display: block; font-size: 13px; font-weight: 600; color: var(--text2); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.form-row { display: flex; gap: 16px; align-items: flex-end; }
.form-row > * { flex: 1; }

input[type="text"], input[type="number"] {
    width: 100%; padding: 10px 14px; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; color: var(--text); font-size: 14px; outline: none; transition: border 0.15s;
}
input:focus { border-color: var(--accent); }
input::placeholder { color: #555; }

.btn { padding: 10px 20px; border-radius: 8px; border: none; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.15s; }
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover { background: #5a52e0; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover { border-color: var(--accent); }
.btn-sm { padding: 6px 14px; font-size: 13px; }
.btn-danger { background: var(--red); color: white; }
.btn-danger:hover { background: #d32f2f; }

/* Range slider */
.range-group { display: flex; align-items: center; gap: 12px; }
.range-group input[type="range"] { flex: 1; accent-color: var(--accent); }
.range-value { font-family: monospace; font-size: 14px; min-width: 50px; text-align: right; color: var(--accent2); }

/* Checkbox */
.checkbox-group { display: flex; align-items: center; gap: 8px; cursor: pointer; }
.checkbox-group input { accent-color: var(--accent); width: 18px; height: 18px; }

/* Cards */
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 16px; }
.card h3 { font-size: 16px; margin-bottom: 12px; }

/* Mask editor */
.mask-editor { position: relative; display: inline-block; cursor: crosshair; border: 2px solid var(--border); border-radius: 8px; overflow: hidden; }
.mask-editor img { display: block; max-width: 100%; }
.mask-editor canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

.mask-list { margin-top: 12px; }
.mask-item { display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: var(--surface2); border-radius: 6px; margin-bottom: 6px; font-size: 13px; font-family: monospace; }
.mask-item .mask-remove { cursor: pointer; color: var(--red); font-weight: bold; margin-left: auto; }

/* Progress */
.progress-bar { width: 100%; height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin: 12px 0; }
.progress-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.3s; }

/* Results */
.results-summary { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
.result-stat { background: var(--surface2); padding: 16px 24px; border-radius: 10px; text-align: center; min-width: 120px; }
.result-stat .num { font-size: 32px; font-weight: 700; }
.result-stat .lbl { font-size: 12px; color: var(--text2); margin-top: 2px; }
.result-stat.pass .num { color: var(--green); }
.result-stat.swap .num { color: var(--orange); }
.result-stat.fail .num { color: var(--red); }
.result-stat.total .num { color: var(--accent2); }

.status-msg { padding: 12px 16px; border-radius: 8px; font-size: 14px; margin-bottom: 16px; }
.status-msg.running { background: #1a237e22; border: 1px solid #3f51b5; color: var(--accent2); }
.status-msg.done { background: #1b5e2022; border: 1px solid var(--green); color: var(--green); }
.status-msg.error { background: #b7121222; border: 1px solid var(--red); color: var(--red); }

/* Folder browser modal */
.modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 100; align-items: center; justify-content: center; }
.modal-overlay.active { display: flex; }
.modal { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; width: 600px; max-height: 70vh; display: flex; flex-direction: column; }
.modal-header { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
.modal-header h3 { font-size: 16px; }
.modal-close { cursor: pointer; color: var(--text2); font-size: 20px; background: none; border: none; }
.modal-body { padding: 12px; flex: 1; overflow-y: auto; }
.modal-footer { padding: 12px 20px; border-top: 1px solid var(--border); display: flex; gap: 8px; justify-content: flex-end; }

.browser-path { font-size: 12px; font-family: monospace; color: var(--accent2); padding: 8px 12px; background: var(--bg); border-radius: 6px; margin-bottom: 8px; word-break: break-all; }
.browser-item { display: flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 14px; }
.browser-item:hover { background: var(--surface2); }
.browser-item.dir { color: var(--accent2); }
.browser-item.file { color: var(--text2); }
.browser-icon { font-size: 16px; width: 24px; text-align: center; }
</style>
</head>
<body>
<div class="app">

<div class="app-header">
    <h1>Visual Regression Diff</h1>
    <span class="subtitle">Compare game level screenshots</span>
    <div style="flex:1"></div>
    <div class="tabs">
        <button class="tab active" onclick="showTab('setup')">Setup</button>
        <button class="tab" onclick="showTab('masks')">Mask Editor</button>
        <button class="tab" onclick="showTab('results')">Results</button>
    </div>
</div>

<div class="main">

<!-- ===== SETUP TAB ===== -->
<div class="tab-content active" id="tab-setup">
    <div class="card">
        <h3>Screenshot Folders</h3>
        <div class="form-group">
            <label>Old Screenshots (before refactor)</label>
            <div style="display:flex;gap:8px">
                <input type="text" id="old-dir" placeholder="/path/to/old/screenshots">
                <button class="btn btn-secondary btn-sm" onclick="openBrowser('old-dir')">Browse</button>
            </div>
        </div>
        <div class="form-group">
            <label>New Screenshots (after refactor)</label>
            <div style="display:flex;gap:8px">
                <input type="text" id="new-dir" placeholder="/path/to/new/screenshots">
                <button class="btn btn-secondary btn-sm" onclick="openBrowser('new-dir')">Browse</button>
            </div>
        </div>
        <div class="form-group">
            <label>Output Directory (report will be saved here)</label>
            <div style="display:flex;gap:8px">
                <input type="text" id="output-dir" placeholder="Leave empty for auto (next to old folder)">
                <button class="btn btn-secondary btn-sm" onclick="openBrowser('output-dir')">Browse</button>
            </div>
        </div>
    </div>

    <div class="card">
        <h3>Options</h3>
        <div class="form-row">
            <div class="form-group">
                <label>SSIM Threshold (higher = stricter)</label>
                <div class="range-group">
                    <input type="range" id="threshold" min="0.9" max="1" step="0.001" value="0.995"
                           oninput="document.getElementById('threshold-val').textContent = this.value">
                    <span class="range-value" id="threshold-val">0.995</span>
                </div>
            </div>
            <div class="form-group">
                <label>Workers (parallel processes)</label>
                <div class="range-group">
                    <input type="range" id="workers" min="1" max="16" step="1" value="4"
                           oninput="document.getElementById('workers-val').textContent = this.value">
                    <span class="range-value" id="workers-val">4</span>
                </div>
            </div>
        </div>
        <div class="form-group">
            <label class="checkbox-group">
                <input type="checkbox" id="save-pass">
                Save diff images for passed levels too (uses more disk space)
            </label>
        </div>
        <div class="form-group">
            <label class="checkbox-group">
                <input type="checkbox" id="color-normalize" onchange="toggleColorK()">
                Color-invariant mode (detect color swaps without false positives)
            </label>
        </div>
        <div class="form-group" id="color-k-group" style="display:none">
            <label>Number of color clusters (K)</label>
            <div class="range-group">
                <input type="range" id="color-k" min="4" max="32" step="1" value="16"
                       oninput="document.getElementById('color-k-val').textContent = this.value">
                <span class="range-value" id="color-k-val">16</span>
            </div>
        </div>
    </div>

    <div class="card" id="mask-summary" style="display:none">
        <h3>Active Masks</h3>
        <p id="mask-count-summary" style="color:var(--accent2);font-size:14px;"></p>
    </div>

    <button class="btn btn-primary" id="run-btn" onclick="runComparison()" style="width:100%;padding:14px;font-size:16px;">
        Run Comparison
    </button>

    <div id="run-status" style="margin-top:16px;display:none;">
        <div class="status-msg running" id="status-msg">Processing...</div>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
    </div>
</div>

<!-- ===== MASK EDITOR TAB ===== -->
<div class="tab-content" id="tab-masks">
    <div class="card">
        <h3>Region Editor</h3>
        <p style="color:var(--text2);font-size:14px;margin-bottom:16px;">
            Load a sample screenshot, then use the tools below to define exclusion masks
            and shift corrections.
        </p>

        <div class="form-group">
            <label>Load a sample screenshot</label>
            <div style="display:flex;gap:8px;align-items:center;">
                <button class="btn btn-secondary btn-sm" onclick="loadSampleFromDir()">Load from Old folder</button>
                <span style="color:var(--text2);font-size:13px;">or</span>
                <label class="btn btn-secondary btn-sm" style="margin:0">
                    Upload image
                    <input type="file" accept="image/png" style="display:none" onchange="loadSampleFile(event)">
                </label>
            </div>
        </div>

        <div id="mask-canvas-container" style="display:none;">

            <!-- Tool selector -->
            <div style="display:flex;gap:8px;margin-bottom:12px;align-items:center;">
                <span style="font-size:13px;color:var(--text2);font-weight:600;">Tool:</span>
                <button class="btn btn-sm tool-btn active" id="tool-mask" onclick="setTool('mask')"
                    style="background:rgba(100,100,255,0.2);border:1px solid rgba(100,100,255,0.5);color:#aaf;">
                    Exclusion Mask</button>
                <button class="btn btn-sm tool-btn" id="tool-shift" onclick="setTool('shift')"
                    style="background:rgba(255,165,0,0.15);border:1px solid rgba(255,165,0,0.4);color:var(--orange);">
                    Shift Region</button>
            </div>

            <!-- Tool description -->
            <div id="tool-desc-mask" style="font-size:13px;color:var(--text2);margin-bottom:10px;">
                Draw rectangles to <b style="color:#aaf">ignore</b> regions (animated areas, UI elements).
            </div>
            <div id="tool-desc-shift" style="font-size:13px;color:var(--text2);margin-bottom:10px;display:none;">
                Draw a rectangle on the region that has <b style="color:var(--orange)">shifted position</b> in the new screenshots,
                then enter the X/Y pixel offset.
            </div>

            <div class="mask-editor" id="mask-editor">
                <img id="mask-img" src="" alt="Sample screenshot">
                <canvas id="mask-canvas"></canvas>
            </div>

            <div style="margin-top:12px;display:flex;gap:8px;">
                <button class="btn btn-secondary btn-sm" onclick="undoLast()">Undo Last</button>
                <button class="btn btn-danger btn-sm" onclick="clearAll()">Clear All</button>
                <div style="flex:1"></div>
                <button class="btn btn-primary btn-sm" onclick="saveMaskConfig()">Save Config</button>
            </div>

            <div class="mask-list" id="mask-list"></div>
        </div>
    </div>
</div>

<!-- ===== RESULTS TAB ===== -->
<div class="tab-content" id="tab-results">
    <div id="no-results" class="card" style="text-align:center;padding:60px;">
        <p style="color:var(--text2);font-size:16px;">No results yet. Run a comparison first.</p>
    </div>
    <div id="results-content" style="display:none;"></div>
</div>

</div><!-- main -->
</div><!-- app -->

<!-- Folder browser modal -->
<div class="modal-overlay" id="browser-modal">
    <div class="modal">
        <div class="modal-header">
            <h3>Select Folder</h3>
            <button class="modal-close" onclick="closeBrowser()">&times;</button>
        </div>
        <div class="modal-body">
            <div class="browser-path" id="browser-path"></div>
            <div id="browser-items"></div>
        </div>
        <div class="modal-footer">
            <button class="btn btn-secondary btn-sm" onclick="browserUp()">Up</button>
            <button class="btn btn-primary btn-sm" onclick="selectFolder()">Select This Folder</button>
        </div>
    </div>
</div>

<script>
// ===== Tab switching =====
function showTab(name) {
    document.querySelectorAll('.tab').forEach((t, i) => {
        const tabs = ['setup', 'masks', 'results'];
        t.classList.toggle('active', tabs[i] === name);
    });
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
}

// ===== Folder browser =====
let browserTarget = null;
let browserCurrentPath = '';

function toggleColorK() {
    const checked = document.getElementById('color-normalize').checked;
    document.getElementById('color-k-group').style.display = checked ? '' : 'none';
}

function openBrowser(inputId) {
    browserTarget = inputId;
    const current = document.getElementById(inputId).value;
    browseTo(current || (typeof process !== 'undefined' ? '' : '~'));
    document.getElementById('browser-modal').classList.add('active');
}

function closeBrowser() {
    document.getElementById('browser-modal').classList.remove('active');
}

function selectFolder() {
    document.getElementById(browserTarget).value = browserCurrentPath;
    closeBrowser();
}

async function browseTo(path) {
    try {
        const res = await fetch('/api/browse', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: path || '~'})
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }

        browserCurrentPath = data.path;
        document.getElementById('browser-path').textContent = data.path;

        const container = document.getElementById('browser-items');
        container.innerHTML = '';
        data.items.forEach(item => {
            if (!item.isDir) return;
            const div = document.createElement('div');
            div.className = 'browser-item dir';
            div.innerHTML = `<span class="browser-icon">&#128193;</span> ${item.name}`;
            div.onclick = () => browseTo(item.path);
            container.appendChild(div);
        });

        // Show png count
        const pngCount = data.items.filter(i => i.isPng).length;
        if (pngCount > 0) {
            const info = document.createElement('div');
            info.style.cssText = 'padding:12px;color:var(--accent2);font-size:13px;border-top:1px solid var(--border);margin-top:8px;';
            info.textContent = `${pngCount} PNG file(s) in this folder`;
            container.appendChild(info);
        }
    } catch(e) {
        alert('Error browsing: ' + e.message);
    }
}

function browserUp() {
    const parts = browserCurrentPath.split('/');
    parts.pop();
    browseTo(parts.join('/') || '/');
}

// ===== Region editor (masks + shifts) =====
let masks = [];
let shifts = [];
let currentTool = 'mask';  // 'mask' or 'shift'
let maskImgNatW = 0, maskImgNatH = 0;
let isDrawing = false;
let drawStart = null;
let maskCanvas, maskCtx;

function setTool(tool) {
    currentTool = tool;
    document.getElementById('tool-mask').classList.toggle('active', tool === 'mask');
    document.getElementById('tool-shift').classList.toggle('active', tool === 'shift');
    document.getElementById('tool-desc-mask').style.display = tool === 'mask' ? '' : 'none';
    document.getElementById('tool-desc-shift').style.display = tool === 'shift' ? '' : 'none';
    // Update cursor style
    maskCanvas.style.cursor = tool === 'mask' ? 'crosshair' : 'move';
}

function initMaskCanvas() {
    const img = document.getElementById('mask-img');
    maskCanvas = document.getElementById('mask-canvas');
    maskCtx = maskCanvas.getContext('2d');

    maskCanvas.width = img.naturalWidth;
    maskCanvas.height = img.naturalHeight;
    maskImgNatW = img.naturalWidth;
    maskImgNatH = img.naturalHeight;

    maskCanvas.onmousedown = (e) => {
        isDrawing = true;
        drawStart = getCanvasCoords(e);
    };
    maskCanvas.onmousemove = (e) => {
        if (!isDrawing) return;
        const cur = getCanvasCoords(e);
        drawAll();
        const color = currentTool === 'mask' ? 'rgba(100, 100, 255, 0.8)' : 'rgba(255, 165, 0, 0.8)';
        maskCtx.strokeStyle = color;
        maskCtx.lineWidth = 2;
        maskCtx.setLineDash([6, 3]);
        maskCtx.strokeRect(drawStart.x, drawStart.y, cur.x - drawStart.x, cur.y - drawStart.y);
        maskCtx.setLineDash([]);
    };
    maskCanvas.onmouseup = (e) => {
        if (!isDrawing) return;
        isDrawing = false;
        const end = getCanvasCoords(e);
        const x = Math.min(drawStart.x, end.x);
        const y = Math.min(drawStart.y, end.y);
        const w = Math.abs(end.x - drawStart.x);
        const h = Math.abs(end.y - drawStart.y);
        if (w > 5 && h > 5) {
            if (currentTool === 'mask') {
                masks.push({x: Math.round(x), y: Math.round(y), width: Math.round(w), height: Math.round(h)});
            } else {
                // Prompt for shift values
                const shiftY = prompt('How many pixels is this region shifted DOWN in the new screenshots?\\n(Use negative for up)', '0');
                const shiftX = prompt('How many pixels is this region shifted RIGHT in the new screenshots?\\n(Use negative for left)', '0');
                if (shiftY !== null && shiftX !== null) {
                    shifts.push({
                        x: Math.round(x), y: Math.round(y),
                        width: Math.round(w), height: Math.round(h),
                        shiftX: parseInt(shiftX) || 0, shiftY: parseInt(shiftY) || 0
                    });
                }
            }
        }
        drawAll();
        updateItemList();
    };
}

function getCanvasCoords(e) {
    const rect = maskCanvas.getBoundingClientRect();
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function drawAll() {
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

    // Draw masks (blue)
    masks.forEach((m, i) => {
        maskCtx.fillStyle = 'rgba(100, 100, 255, 0.25)';
        maskCtx.fillRect(m.x, m.y, m.width, m.height);
        maskCtx.strokeStyle = 'rgba(100, 100, 255, 0.8)';
        maskCtx.lineWidth = 2;
        maskCtx.strokeRect(m.x, m.y, m.width, m.height);
        maskCtx.fillStyle = 'rgba(100, 100, 255, 0.9)';
        maskCtx.font = '14px monospace';
        maskCtx.fillText(`Mask ${i + 1}`, m.x + 4, m.y + 16);
    });

    // Draw shifts (orange)
    shifts.forEach((s, i) => {
        maskCtx.fillStyle = 'rgba(255, 165, 0, 0.2)';
        maskCtx.fillRect(s.x, s.y, s.width, s.height);
        maskCtx.strokeStyle = 'rgba(255, 165, 0, 0.8)';
        maskCtx.lineWidth = 2;
        maskCtx.strokeRect(s.x, s.y, s.width, s.height);
        maskCtx.fillStyle = 'rgba(255, 165, 0, 0.95)';
        maskCtx.font = '14px monospace';
        maskCtx.fillText(`Shift ${i + 1}`, s.x + 4, s.y + 16);
        // Draw arrow showing shift direction
        const cx = s.x + s.width / 2;
        const cy = s.y + s.height / 2;
        const ax = cx + s.shiftX * 0.5;
        const ay = cy + s.shiftY * 0.5;
        maskCtx.strokeStyle = 'rgba(255, 165, 0, 0.9)';
        maskCtx.lineWidth = 3;
        maskCtx.beginPath();
        maskCtx.moveTo(cx, cy);
        maskCtx.lineTo(ax, ay);
        maskCtx.stroke();
        // Arrowhead
        const angle = Math.atan2(s.shiftY, s.shiftX);
        maskCtx.beginPath();
        maskCtx.moveTo(ax, ay);
        maskCtx.lineTo(ax - 10 * Math.cos(angle - 0.4), ay - 10 * Math.sin(angle - 0.4));
        maskCtx.moveTo(ax, ay);
        maskCtx.lineTo(ax - 10 * Math.cos(angle + 0.4), ay - 10 * Math.sin(angle + 0.4));
        maskCtx.stroke();
    });

    updateSummary();
}

function updateItemList() {
    const container = document.getElementById('mask-list');
    container.innerHTML = '';

    masks.forEach((m, i) => {
        const div = document.createElement('div');
        div.className = 'mask-item';
        div.style.borderLeft = '3px solid rgba(100,100,255,0.8)';
        div.innerHTML = `
            <span style="color:#aaf;">Mask ${i + 1}</span>
            <span>x:${m.x} y:${m.y} w:${m.width} h:${m.height}</span>
            <span class="mask-remove" onclick="removeMask(${i})">&times;</span>
        `;
        container.appendChild(div);
    });

    shifts.forEach((s, i) => {
        const div = document.createElement('div');
        div.className = 'mask-item';
        div.style.borderLeft = '3px solid rgba(255,165,0,0.8)';
        div.innerHTML = `
            <span style="color:var(--orange);">Shift ${i + 1}</span>
            <span>x:${s.x} y:${s.y} w:${s.width} h:${s.height}</span>
            <span style="color:var(--orange);font-weight:bold;">dx:${s.shiftX} dy:${s.shiftY}</span>
            <span class="mask-remove" onclick="removeShift(${i})">&times;</span>
        `;
        container.appendChild(div);
    });
}

function updateSummary() {
    const el = document.getElementById('mask-summary');
    const countEl = document.getElementById('mask-count-summary');
    const parts = [];
    if (masks.length > 0) parts.push(`${masks.length} mask(s)`);
    if (shifts.length > 0) parts.push(`${shifts.length} shift correction(s)`);
    if (parts.length > 0) {
        el.style.display = '';
        countEl.textContent = parts.join(' and ') + ' configured';
    } else {
        el.style.display = 'none';
    }
}

function removeMask(i) { masks.splice(i, 1); drawAll(); updateItemList(); }
function removeShift(i) { shifts.splice(i, 1); drawAll(); updateItemList(); }
function undoLast() {
    if (currentTool === 'mask' && masks.length > 0) masks.pop();
    else if (currentTool === 'shift' && shifts.length > 0) shifts.pop();
    else if (shifts.length > 0) shifts.pop();
    else if (masks.length > 0) masks.pop();
    drawAll(); updateItemList();
}
function clearAll() { masks = []; shifts = []; drawAll(); updateItemList(); }

async function loadSampleFromDir() {
    const dir = document.getElementById('old-dir').value;
    if (!dir) { alert('Set the Old Screenshots folder first'); return; }
    try {
        const res = await fetch('/api/sample-image', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: dir})
        });
        if (!res.ok) { alert('No PNG found in that folder'); return; }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        loadImage(url);
    } catch(e) { alert('Error: ' + e.message); }
}

function loadSampleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadImage(url);
}

function loadImage(url) {
    const img = document.getElementById('mask-img');
    img.onload = () => {
        document.getElementById('mask-canvas-container').style.display = '';
        initMaskCanvas();
        drawAll();
    };
    img.src = url;
}

async function saveMaskConfig() {
    const path = prompt('Save config as:', 'mask_config.json');
    if (!path) return;
    try {
        await fetch('/api/save-mask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path, masks, shifts, imageWidth: maskImgNatW, imageHeight: maskImgNatH})
        });
        alert('Saved: ' + path);
    } catch(e) { alert('Error: ' + e.message); }
}

// ===== Run comparison =====
async function runComparison() {
    const oldDir = document.getElementById('old-dir').value;
    const newDir = document.getElementById('new-dir').value;
    if (!oldDir || !newDir) { alert('Please set both screenshot folders'); return; }

    const hasRegions = masks.length > 0 || shifts.length > 0;
    const payload = {
        oldDir: oldDir,
        newDir: newDir,
        outputDir: document.getElementById('output-dir').value,
        threshold: parseFloat(document.getElementById('threshold').value),
        workers: parseInt(document.getElementById('workers').value),
        savePass: document.getElementById('save-pass').checked,
        colorNormalizeK: document.getElementById('color-normalize').checked
                         ? parseInt(document.getElementById('color-k').value) : 0,
        masks: masks.length > 0 ? masks : [],
        shifts: shifts.length > 0 ? shifts : [],
        maskImageSize: hasRegions ? [maskImgNatW, maskImgNatH] : null,
    };

    document.getElementById('run-btn').disabled = true;
    document.getElementById('run-status').style.display = '';
    document.getElementById('status-msg').className = 'status-msg running';
    document.getElementById('status-msg').textContent = 'Starting...';

    try {
        await fetch('/api/run', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        pollStatus();
    } catch(e) {
        document.getElementById('status-msg').className = 'status-msg error';
        document.getElementById('status-msg').textContent = 'Error: ' + e.message;
        document.getElementById('run-btn').disabled = false;
    }
}

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('status-msg').textContent = data.message || data.status;

        if (data.status === 'running') {
            if (data.total > 0) {
                const pct = (data.progress / data.total * 100).toFixed(0);
                document.getElementById('progress-fill').style.width = pct + '%';
            }
            setTimeout(pollStatus, 500);
        } else if (data.status === 'done') {
            document.getElementById('status-msg').className = 'status-msg done';
            document.getElementById('progress-fill').style.width = '100%';
            document.getElementById('run-btn').disabled = false;
            showResults(data.results);
            showTab('results');
        } else if (data.status === 'error') {
            document.getElementById('status-msg').className = 'status-msg error';
            document.getElementById('run-btn').disabled = false;
        }
    } catch(e) {
        setTimeout(pollStatus, 1000);
    }
}

function showResults(results) {
    document.getElementById('no-results').style.display = 'none';
    const container = document.getElementById('results-content');
    container.style.display = '';

    const swapCard = results.swapped ? `<div class="result-stat swap"><div class="num">${results.swapped}</div><div class="lbl">Color Swaps</div></div>` : '';
    let html = `
        <div class="results-summary">
            <div class="result-stat total"><div class="num">${results.total}</div><div class="lbl">Total</div></div>
            <div class="result-stat pass"><div class="num">${results.passed}</div><div class="lbl">Passed</div></div>
            ${swapCard}
            <div class="result-stat fail"><div class="num">${results.failed}</div><div class="lbl">Failed</div></div>
        </div>
    `;

    if (results.report_path) {
        html += `<div class="card" style="text-align:center;">
            <p style="margin-bottom:12px;">Full interactive report generated:</p>
            <p style="font-family:monospace;color:var(--accent2);word-break:break-all;">${results.report_path}</p>
            <p style="margin-top:12px;color:var(--text2);font-size:13px;">Open this file in your browser to view the detailed report with diff images.</p>
        </div>`;
    }

    if (results.levels) {
        const failed = results.levels.filter(l => !l.passed);
        const swappedLevels = results.levels.filter(l => l.passed && l.color_swapped);
        if (failed.length > 0) {
            html += '<div class="card"><h3>Failed Levels</h3>';
            failed.forEach(l => {
                const pct = (l.score * 100).toFixed(2);
                html += `<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);">
                    <span style="color:var(--red);font-weight:700;font-size:12px;">FAIL</span>
                    <span style="font-weight:600;min-width:120px;">${l.level}</span>
                    <span style="color:var(--text2);font-family:monospace;font-size:13px;">SSIM: ${l.score.toFixed(6)}</span>
                    <span style="color:var(--orange);font-size:13px;">${l.change_pct}% changed</span>
                    <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden;max-width:200px;">
                        <div style="height:100%;width:${pct}%;background:var(--red);border-radius:3px;"></div>
                    </div>
                </div>`;
            });
            html += '</div>';
        }
        if (swappedLevels.length > 0) {
            html += '<div class="card"><h3>Color Swaps (distribution unchanged)</h3>';
            swappedLevels.forEach(l => {
                const pct = (l.score * 100).toFixed(2);
                html += `<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);">
                    <span style="color:var(--orange);font-weight:700;font-size:12px;">SWAP</span>
                    <span style="font-weight:600;min-width:120px;">${l.level}</span>
                    <span style="color:var(--text2);font-family:monospace;font-size:13px;">SSIM: ${l.score.toFixed(6)}</span>
                    <span style="color:var(--orange);font-size:13px;">${l.change_pct}% changed</span>
                    <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden;max-width:200px;">
                        <div style="height:100%;width:${pct}%;background:var(--orange);border-radius:3px;"></div>
                    </div>
                </div>`;
            });
            html += '</div>';
        }
    }

    container.innerHTML = html;
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
    print(f"\n  Visual Regression Diff Tool")
    print(f"  Open in browser: http://localhost:{port}\n")
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
