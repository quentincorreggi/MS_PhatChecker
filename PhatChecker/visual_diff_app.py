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
import color_pattern

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


@app.route("/api/sample-image-path", methods=["POST"])
def sample_image_path():
    """Like /api/sample-image but returns the server-side path (used for preview)."""
    data = request.json
    path = data.get("path", "")
    if not os.path.isdir(path):
        return jsonify({"error": "Not a directory"}), 400
    for f in sorted(os.listdir(path)):
        if f.lower().endswith(".png"):
            return jsonify({"path": os.path.join(path, f)})
    return jsonify({"error": "No PNG files found"}), 404


@app.route("/api/serve-image")
def serve_image():
    """Serve an arbitrary local PNG (used by Grid Editor to display the sample)."""
    path = request.args.get("path", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/png")


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
    masks = data.get("masks", [])
    shifts = data.get("shifts", [])
    mask_image_size = data.get("maskImageSize", None)
    grid_config = data.get("gridConfig", None)
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
            gc = grid_config if grid_config and grid_config.get("grids") else None

            results, only_old, only_new = run_batch(
                old_dir, new_dir, output_dir, threshold, workers, save_pass,
                mask_list, msize, shift_list, gc
            )

            report_path = generate_html_report(results, only_old, only_new, output_dir, threshold)

            failed = [r for r in results if not r["passed"]]
            passed_list = [r for r in results if r["passed"]]
            pattern_failed = [r for r in results if r.get("pattern") and not r["pattern"]["passed"]]

            current_job = {
                "status": "done",
                "progress": len(results),
                "total": len(results),
                "message": f"Done! {len(passed_list)} passed, {len(failed)} failed"
                           + (f", {len(pattern_failed)} pattern-fail" if gc else ""),
                "results": {
                    "total": len(results),
                    "passed": len(passed_list),
                    "failed": len(failed),
                    "pattern_enabled": bool(gc),
                    "pattern_failed": len(pattern_failed),
                    "missing_old": len(only_old),
                    "missing_new": len(only_new),
                    "report_path": report_path,
                    "output_dir": output_dir,
                    "levels": [
                        {"level": r["level"], "score": float(r["score"]),
                         "passed": bool(r["passed"]),
                         "change_pct": float(r["change_pct"]),
                         "pattern_passed": (None if not r.get("pattern")
                                            else bool(r["pattern"]["passed"])),
                         "pattern_mismatches": (0 if not r.get("pattern")
                                                else int(r["pattern"]["mismatches"]))}
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
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return jsonify({"saved": save_path})


@app.route("/api/save-grid", methods=["POST"])
def save_grid():
    """Save grid config (color pattern calibration) to a JSON file."""
    data = request.json
    save_path = data.get("path", "grid_config.json")
    config = {
        "imageWidth": data.get("imageWidth", 0),
        "imageHeight": data.get("imageHeight", 0),
        "grids": data.get("grids", []),
        "cellInsetPct": data.get("cellInsetPct", 0.20),
        "satThreshold": data.get("satThreshold", 0.25),
        "emptyCoveragePct": data.get("emptyCoveragePct", 0.08),
        "minK": data.get("minK", 2),
        "maxK": data.get("maxK", 8),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return jsonify({"saved": save_path})


@app.route("/api/load-grid", methods=["POST"])
def load_grid():
    """Load a saved grid config (so the user can re-edit it)."""
    data = request.json
    path = data.get("path", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/preview-pattern", methods=["POST"])
def preview_pattern():
    """Run pattern analysis on a single image and return the label grids.
    Used by the Grid Editor to preview the clustering result.
    """
    data = request.json
    img_path = data.get("imagePath")
    config_dict = data.get("gridConfig")
    if not img_path or not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 400
    if not config_dict or not config_dict.get("grids"):
        return jsonify({"error": "Grid config required"}), 400
    try:
        config = color_pattern.GridConfig.from_dict(config_dict)
        pattern = color_pattern.analyze(img_path, config)
        return jsonify({
            "grids": pattern.grids,
            "palette": pattern.palette,
            "k": pattern.k,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
.result-stat.fail .num { color: var(--red); }
.result-stat.total .num { color: var(--accent2); }

.status-msg { padding: 12px 16px; border-radius: 8px; font-size: 14px; margin-bottom: 16px; }
.status-msg.running { background: #1a237e22; border: 1px solid #3f51b5; color: var(--accent2); }
.status-msg.done { background: #1b5e2022; border: 1px solid var(--green); color: var(--green); }
.status-msg.error { background: #b7121222; border: 1px solid var(--red); color: var(--red); }

.filter-btn { background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 6px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; }
.filter-btn:hover { border-color: var(--accent); }
.filter-btn.active { background: #0f3460; border-color: var(--accent2); color: var(--accent2); }

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
        <button class="tab" onclick="showTab('grids')">Grid Editor</button>
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
        <div class="form-group">
            <label>Grid Config (optional — enables color-agnostic pattern check)</label>
            <div style="display:flex;gap:8px">
                <input type="text" id="grid-config-path" placeholder="Leave empty to skip pattern check, or path to grid_config.json">
                <button class="btn btn-secondary btn-sm" onclick="showTab('grids')">Edit</button>
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
    </div>

    <div class="card" id="mask-summary" style="display:none">
        <h3>Active Masks</h3>
        <p id="mask-count-summary" style="color:var(--accent2);font-size:14px;"></p>
    </div>

    <div class="card" id="grid-summary" style="display:none">
        <h3>Color Pattern Comparison</h3>
        <p id="grid-count-summary" style="color:var(--accent2);font-size:14px;"></p>
        <p style="color:var(--text2);font-size:13px;margin-top:6px;">
            Each pair will also be checked for color-agnostic pattern equivalence
            using the calibrated grids. The pattern verdict is reported alongside SSIM.
        </p>
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

<!-- ===== GRID EDITOR TAB ===== -->
<div class="tab-content" id="tab-grids">
    <div class="card">
        <h3>Color Pattern Grid Calibration</h3>
        <p style="color:var(--text2);font-size:14px;margin-bottom:14px;">
            Draw the two ingredient grids on a sample screenshot. The tool will extract a
            color-agnostic label pattern from each cell and compare patterns between
            old &amp; new screenshots — so a re-colored level (same layout, different
            palette) still passes.
        </p>

        <div class="form-group">
            <label>Load a sample screenshot</label>
            <div style="display:flex;gap:8px;align-items:center;">
                <button class="btn btn-secondary btn-sm" onclick="loadGridSampleFromDir()">Load from Old folder</button>
                <button class="btn btn-secondary btn-sm" onclick="loadGridConfig()">Load saved config…</button>
            </div>
        </div>

        <div id="grid-canvas-container" style="display:none;">
            <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;align-items:flex-end;">
                <div>
                    <label style="display:block;font-size:12px;color:var(--text2);">Grid name</label>
                    <input type="text" id="grid-name" value="top" style="width:120px;">
                </div>
                <div>
                    <label style="display:block;font-size:12px;color:var(--text2);">Rows</label>
                    <input type="number" id="grid-rows" value="7" min="1" max="20" style="width:80px;">
                </div>
                <div>
                    <label style="display:block;font-size:12px;color:var(--text2);">Cols</label>
                    <input type="number" id="grid-cols" value="7" min="1" max="20" style="width:80px;">
                </div>
                <div style="font-size:13px;color:var(--text2);">
                    Set rows/cols → then drag a rectangle around that grid on the image.
                </div>
            </div>

            <div class="mask-editor" id="grid-editor">
                <img id="grid-img" src="" alt="Sample screenshot">
                <canvas id="grid-canvas"></canvas>
            </div>

            <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
                <button class="btn btn-secondary btn-sm" onclick="undoLastGrid()">Undo Last Grid</button>
                <button class="btn btn-danger btn-sm" onclick="clearAllGrids()">Clear All</button>
                <div style="flex:1"></div>
                <button class="btn btn-secondary btn-sm" onclick="runGridPreview()">Preview Pattern</button>
                <button class="btn btn-primary btn-sm" onclick="saveGridConfig()">Save Config</button>
            </div>

            <details style="margin-top:12px;">
                <summary style="cursor:pointer;color:var(--accent2);font-size:13px;">Advanced sampling options</summary>
                <div style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap;">
                    <div>
                        <label style="display:block;font-size:12px;color:var(--text2);">Cell inset</label>
                        <input type="number" id="grid-inset" value="0.20" min="0" max="0.45" step="0.01" style="width:80px;">
                        <div style="font-size:11px;color:var(--text2);">Inner sample fraction</div>
                    </div>
                    <div>
                        <label style="display:block;font-size:12px;color:var(--text2);">Sat threshold</label>
                        <input type="number" id="grid-sat" value="0.25" min="0" max="1" step="0.01" style="width:80px;">
                        <div style="font-size:11px;color:var(--text2);">Min saturation = ingredient pixel</div>
                    </div>
                    <div>
                        <label style="display:block;font-size:12px;color:var(--text2);">Empty coverage</label>
                        <input type="number" id="grid-empty" value="0.08" min="0" max="1" step="0.01" style="width:80px;">
                        <div style="font-size:11px;color:var(--text2);">Below = empty cell</div>
                    </div>
                    <div>
                        <label style="display:block;font-size:12px;color:var(--text2);">Max colors (k)</label>
                        <input type="number" id="grid-maxk" value="8" min="2" max="12" step="1" style="width:80px;">
                    </div>
                </div>
            </details>

            <div class="mask-list" id="grid-list"></div>

            <div id="grid-preview-result" style="margin-top:12px;display:none;"></div>
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
        const tabs = ['setup', 'masks', 'grids', 'results'];
        t.classList.toggle('active', tabs[i] === name);
    });
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
}

// ===== Folder browser =====
let browserTarget = null;
let browserCurrentPath = '';

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

// ===== Grid Editor =====
let gridSpecs = [];                   // [{name, rows, cols, x, y, width, height}]
let gridImgPath = '';                 // server-side path for preview API
let gridImgNatW = 0, gridImgNatH = 0;
let gridCanvas, gridCtx;
let gridIsDrawing = false;
let gridDrawStart = null;

function initGridCanvas() {
    const img = document.getElementById('grid-img');
    gridCanvas = document.getElementById('grid-canvas');
    gridCtx = gridCanvas.getContext('2d');
    gridCanvas.width = img.naturalWidth;
    gridCanvas.height = img.naturalHeight;
    gridImgNatW = img.naturalWidth;
    gridImgNatH = img.naturalHeight;

    gridCanvas.onmousedown = (e) => {
        gridIsDrawing = true;
        gridDrawStart = getGridCanvasCoords(e);
    };
    gridCanvas.onmousemove = (e) => {
        if (!gridIsDrawing) return;
        const cur = getGridCanvasCoords(e);
        drawAllGrids();
        gridCtx.strokeStyle = 'rgba(108, 99, 255, 0.9)';
        gridCtx.lineWidth = 2;
        gridCtx.setLineDash([6, 3]);
        gridCtx.strokeRect(gridDrawStart.x, gridDrawStart.y,
                           cur.x - gridDrawStart.x, cur.y - gridDrawStart.y);
        gridCtx.setLineDash([]);
    };
    gridCanvas.onmouseup = (e) => {
        if (!gridIsDrawing) return;
        gridIsDrawing = false;
        const end = getGridCanvasCoords(e);
        const x = Math.min(gridDrawStart.x, end.x);
        const y = Math.min(gridDrawStart.y, end.y);
        const w = Math.abs(end.x - gridDrawStart.x);
        const h = Math.abs(end.y - gridDrawStart.y);
        if (w > 20 && h > 20) {
            const name = document.getElementById('grid-name').value.trim() || `grid${gridSpecs.length + 1}`;
            const rows = parseInt(document.getElementById('grid-rows').value) || 7;
            const cols = parseInt(document.getElementById('grid-cols').value) || 7;
            gridSpecs.push({name, rows, cols,
                            x: Math.round(x), y: Math.round(y),
                            width: Math.round(w), height: Math.round(h)});
            // Suggest defaults for the next grid
            if (gridSpecs.length === 1) {
                document.getElementById('grid-name').value = 'bottom';
                document.getElementById('grid-rows').value = 5;
                document.getElementById('grid-cols').value = 7;
            }
        }
        drawAllGrids();
        updateGridList();
        updateGridSummary();
    };
}

function getGridCanvasCoords(e) {
    const rect = gridCanvas.getBoundingClientRect();
    const scaleX = gridCanvas.width / rect.width;
    const scaleY = gridCanvas.height / rect.height;
    return {x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY};
}

function drawAllGrids() {
    gridCtx.clearRect(0, 0, gridCanvas.width, gridCanvas.height);
    gridSpecs.forEach((g, i) => {
        gridCtx.fillStyle = 'rgba(108, 99, 255, 0.10)';
        gridCtx.fillRect(g.x, g.y, g.width, g.height);
        gridCtx.strokeStyle = 'rgba(108, 99, 255, 0.95)';
        gridCtx.lineWidth = 2;
        gridCtx.strokeRect(g.x, g.y, g.width, g.height);
        // Cell lines
        const cw = g.width / g.cols;
        const ch = g.height / g.rows;
        gridCtx.strokeStyle = 'rgba(108, 99, 255, 0.4)';
        gridCtx.lineWidth = 1;
        for (let c = 1; c < g.cols; c++) {
            gridCtx.beginPath();
            gridCtx.moveTo(g.x + c * cw, g.y);
            gridCtx.lineTo(g.x + c * cw, g.y + g.height);
            gridCtx.stroke();
        }
        for (let r = 1; r < g.rows; r++) {
            gridCtx.beginPath();
            gridCtx.moveTo(g.x, g.y + r * ch);
            gridCtx.lineTo(g.x + g.width, g.y + r * ch);
            gridCtx.stroke();
        }
        gridCtx.fillStyle = 'rgba(108, 99, 255, 0.95)';
        gridCtx.font = '14px monospace';
        gridCtx.fillText(`${g.name} (${g.rows}x${g.cols})`, g.x + 4, g.y + 16);
    });
}

function updateGridList() {
    const container = document.getElementById('grid-list');
    container.innerHTML = '';
    gridSpecs.forEach((g, i) => {
        const div = document.createElement('div');
        div.className = 'mask-item';
        div.style.borderLeft = '3px solid rgba(108,99,255,0.8)';
        div.innerHTML = `
            <span style="color:#aab;">${g.name}</span>
            <span>${g.rows}x${g.cols}</span>
            <span>x:${g.x} y:${g.y} w:${g.width} h:${g.height}</span>
            <span class="mask-remove" onclick="removeGrid(${i})">&times;</span>
        `;
        container.appendChild(div);
    });
}

function updateGridSummary() {
    const el = document.getElementById('grid-summary');
    const countEl = document.getElementById('grid-count-summary');
    if (gridSpecs.length > 0) {
        el.style.display = '';
        const names = gridSpecs.map(g => `${g.name} ${g.rows}x${g.cols}`).join(', ');
        countEl.textContent = `${gridSpecs.length} grid(s): ${names}`;
    } else {
        el.style.display = 'none';
    }
}

function removeGrid(i) { gridSpecs.splice(i, 1); drawAllGrids(); updateGridList(); updateGridSummary(); }
function undoLastGrid() { if (gridSpecs.length) gridSpecs.pop(); drawAllGrids(); updateGridList(); updateGridSummary(); }
function clearAllGrids() { gridSpecs = []; drawAllGrids(); updateGridList(); updateGridSummary(); document.getElementById('grid-preview-result').style.display = 'none'; }

async function loadGridSampleFromDir() {
    const dir = document.getElementById('old-dir').value;
    if (!dir) { alert('Set the Old Screenshots folder first'); return; }
    try {
        const res = await fetch('/api/sample-image-path', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path: dir})
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }
        gridImgPath = data.path;
        loadGridImage('/api/serve-image?path=' + encodeURIComponent(data.path));
    } catch(e) { alert('Error: ' + e.message); }
}

function loadGridImage(url) {
    const img = document.getElementById('grid-img');
    img.onload = () => {
        document.getElementById('grid-canvas-container').style.display = '';
        initGridCanvas();
        drawAllGrids();
    };
    img.src = url;
}

function currentGridConfigObj() {
    return {
        imageWidth: gridImgNatW,
        imageHeight: gridImgNatH,
        grids: gridSpecs,
        cellInsetPct: parseFloat(document.getElementById('grid-inset').value) || 0.20,
        satThreshold: parseFloat(document.getElementById('grid-sat').value) || 0.25,
        emptyCoveragePct: parseFloat(document.getElementById('grid-empty').value) || 0.08,
        minK: 2,
        maxK: parseInt(document.getElementById('grid-maxk').value) || 8,
    };
}

async function runGridPreview() {
    if (!gridImgPath) { alert('Load a sample image first'); return; }
    if (gridSpecs.length === 0) { alert('Draw at least one grid first'); return; }
    const result = document.getElementById('grid-preview-result');
    result.style.display = '';
    result.innerHTML = '<div style="color:var(--text2);">Analyzing…</div>';
    try {
        const res = await fetch('/api/preview-pattern', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({imagePath: gridImgPath, gridConfig: currentGridConfigObj()})
        });
        const data = await res.json();
        if (data.error) { result.innerHTML = `<div class="error">${data.error}</div>`; return; }
        let html = `<div style="color:var(--accent2);margin-bottom:8px;">Detected ${data.k} color(s).</div>`;
        for (const [name, grid] of Object.entries(data.grids)) {
            html += `<div style="margin-bottom:10px;"><b>${name}</b><pre style="background:var(--bg);padding:10px;border-radius:6px;font-family:monospace;font-size:13px;line-height:1.4;">`;
            html += grid.map(row => row.join(' ')).join('\n');
            html += `</pre></div>`;
        }
        if (data.palette && Object.keys(data.palette).length) {
            html += `<div style="display:flex;gap:8px;flex-wrap:wrap;">`;
            for (const [lbl, info] of Object.entries(data.palette)) {
                const rgb = info.rgb || [128,128,128];
                html += `<div style="display:flex;align-items:center;gap:6px;padding:6px 10px;background:var(--surface2);border-radius:6px;font-size:13px;">
                    <span style="display:inline-block;width:18px;height:18px;border-radius:4px;background:rgb(${rgb[0]},${rgb[1]},${rgb[2]});border:1px solid #333;"></span>
                    <span><b>${lbl}</b> · ${info.count} cells</span>
                </div>`;
            }
            html += `</div>`;
        }
        result.innerHTML = html;
    } catch(e) {
        result.innerHTML = `<div class="error">${e.message}</div>`;
    }
}

async function saveGridConfig() {
    if (gridSpecs.length === 0) { alert('Draw at least one grid first'); return; }
    const path = prompt('Save grid config as:', 'grid_config.json');
    if (!path) return;
    const payload = {path, ...currentGridConfigObj()};
    try {
        await fetch('/api/save-grid', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        document.getElementById('grid-config-path').value = path;
        alert('Saved: ' + path);
    } catch(e) { alert('Error: ' + e.message); }
}

async function loadGridConfig() {
    const path = prompt('Load grid config from:', 'grid_config.json');
    if (!path) return;
    try {
        const res = await fetch('/api/load-grid', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path})
        });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }
        gridImgNatW = data.imageWidth || 0;
        gridImgNatH = data.imageHeight || 0;
        gridSpecs = data.grids || [];
        if (data.cellInsetPct !== undefined) document.getElementById('grid-inset').value = data.cellInsetPct;
        if (data.satThreshold !== undefined) document.getElementById('grid-sat').value = data.satThreshold;
        if (data.emptyCoveragePct !== undefined) document.getElementById('grid-empty').value = data.emptyCoveragePct;
        if (data.maxK !== undefined) document.getElementById('grid-maxk').value = data.maxK;
        document.getElementById('grid-config-path').value = path;
        updateGridList();
        updateGridSummary();
        // If a sample was already loaded, redraw onto it.
        if (gridCanvas) drawAllGrids();
        alert('Loaded ' + (gridSpecs.length) + ' grid(s) from ' + path);
    } catch(e) { alert('Error: ' + e.message); }
}

// ===== Run comparison =====
async function runComparison() {
    const oldDir = document.getElementById('old-dir').value;
    const newDir = document.getElementById('new-dir').value;
    if (!oldDir || !newDir) { alert('Please set both screenshot folders'); return; }

    const hasRegions = masks.length > 0 || shifts.length > 0;
    let gridConfig = null;
    const gridConfigPath = document.getElementById('grid-config-path').value.trim();
    if (gridConfigPath) {
        try {
            const res = await fetch('/api/load-grid', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: gridConfigPath})
            });
            const data = await res.json();
            if (data.error) { alert('Grid config load failed: ' + data.error); return; }
            gridConfig = data;
        } catch(e) { alert('Grid config load failed: ' + e.message); return; }
    } else if (gridSpecs.length > 0 && gridImgNatW > 0) {
        gridConfig = currentGridConfigObj();
    }
    const payload = {
        oldDir: oldDir,
        newDir: newDir,
        outputDir: document.getElementById('output-dir').value,
        threshold: parseFloat(document.getElementById('threshold').value),
        workers: parseInt(document.getElementById('workers').value),
        savePass: document.getElementById('save-pass').checked,
        masks: masks.length > 0 ? masks : [],
        shifts: shifts.length > 0 ? shifts : [],
        maskImageSize: hasRegions ? [maskImgNatW, maskImgNatH] : null,
        gridConfig: gridConfig,
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

let resultsFilter = 'any-fail';
let lastResults = null;

function setResultsFilter(filter, btn) {
    resultsFilter = filter;
    document.querySelectorAll('.results-filter-btn').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');
    if (lastResults) renderLevels(lastResults);
}

function renderLevels(results) {
    const list = document.getElementById('levels-list');
    if (!list) return;
    const levels = results.levels || [];
    const filtered = levels.filter(l => {
        const ssimFail = !l.passed;
        const patternFail = l.pattern_passed === false;
        const anyFail = ssimFail || patternFail;
        if (resultsFilter === 'all') return true;
        if (resultsFilter === 'any-fail') return anyFail;
        if (resultsFilter === 'ssim-fail') return ssimFail;
        if (resultsFilter === 'pattern-fail') return patternFail;
        if (resultsFilter === 'pass') return !anyFail;
        return true;
    });

    if (filtered.length === 0) {
        list.innerHTML = '<div style="padding:20px;color:var(--text2);text-align:center;">No levels match this filter.</div>';
        return;
    }

    list.innerHTML = filtered.map(l => {
        const pct = (l.score * 100).toFixed(2);
        const ssimBadge = l.passed
            ? '<span style="color:var(--green);font-weight:700;font-size:11px;">SSIM OK</span>'
            : '<span style="color:var(--red);font-weight:700;font-size:11px;">SSIM FAIL</span>';
        let patternBadge = '';
        if (l.pattern_passed === true) {
            patternBadge = '<span style="color:var(--accent2);font-weight:700;font-size:11px;">PATTERN OK</span>';
        } else if (l.pattern_passed === false) {
            patternBadge = `<span style="color:var(--orange);font-weight:700;font-size:11px;">PATTERN FAIL (${l.pattern_mismatches})</span>`;
        }
        return `<div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid var(--border);flex-wrap:wrap;">
            ${ssimBadge}
            ${patternBadge}
            <span style="font-weight:600;min-width:120px;">${l.level}</span>
            <span style="color:var(--text2);font-family:monospace;font-size:13px;">SSIM: ${l.score.toFixed(6)}</span>
            <span style="color:var(--orange);font-size:13px;">${l.change_pct}% changed</span>
            <div style="flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden;max-width:200px;">
                <div style="height:100%;width:${pct}%;background:${l.passed ? 'var(--green)' : 'var(--red)'};border-radius:3px;"></div>
            </div>
        </div>`;
    }).join('');
}

function showResults(results) {
    lastResults = results;
    document.getElementById('no-results').style.display = 'none';
    const container = document.getElementById('results-content');
    container.style.display = '';

    const patternEnabled = !!results.pattern_enabled;
    const patternFailed = results.pattern_failed || 0;

    let html = `
        <div class="results-summary">
            <div class="result-stat total"><div class="num">${results.total}</div><div class="lbl">Total</div></div>
            <div class="result-stat pass"><div class="num">${results.passed}</div><div class="lbl">SSIM Passed</div></div>
            <div class="result-stat fail"><div class="num">${results.failed}</div><div class="lbl">SSIM Failed</div></div>
            ${patternEnabled ? `<div class="result-stat fail"><div class="num">${patternFailed}</div><div class="lbl">Pattern Failed</div></div>` : ''}
        </div>
    `;

    if (results.report_path) {
        html += `<div class="card" style="text-align:center;">
            <p style="margin-bottom:12px;">Full interactive report generated:</p>
            <p style="font-family:monospace;color:var(--accent2);word-break:break-all;">${results.report_path}</p>
            <p style="margin-top:12px;color:var(--text2);font-size:13px;">Open this file in your browser to view the detailed report with diff images.</p>
        </div>`;
    }

    html += `<div class="card">
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px;">
            <h3 style="margin:0;flex:1;">Levels</h3>
            <button class="filter-btn results-filter-btn" onclick="setResultsFilter('all', this)">All</button>
            <button class="filter-btn results-filter-btn active" onclick="setResultsFilter('any-fail', this)">Any failure</button>
            <button class="filter-btn results-filter-btn" onclick="setResultsFilter('ssim-fail', this)">SSIM fails only</button>
            ${patternEnabled ? `<button class="filter-btn results-filter-btn" onclick="setResultsFilter('pattern-fail', this)">Pattern fails only</button>` : ''}
            <button class="filter-btn results-filter-btn" onclick="setResultsFilter('pass', this)">Passed only</button>
        </div>
        <div id="levels-list"></div>
    </div>`;

    container.innerHTML = html;
    resultsFilter = 'any-fail';
    renderLevels(results);
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
