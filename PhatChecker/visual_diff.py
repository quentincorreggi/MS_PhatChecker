#!/usr/bin/env python3
"""
Visual Regression Diff Tool for Game Levels
Batch-compares old vs new level screenshots and generates an HTML report.
Supports exclusion masks to ignore animated/dynamic regions.

Usage:
    # Compare two folders of screenshots
    python3 visual_diff.py old_screenshots/ new_screenshots/

    # With a mask config
    python3 visual_diff.py old/ new/ --mask mask_config.json

    # Compare a single pair
    python3 visual_diff.py --single old/level_190.png new/level_190.png
"""

import argparse
import sys
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from scipy.cluster.vq import kmeans2, vq
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Mask & shift support
# ---------------------------------------------------------------------------

def load_mask_config(mask_path):
    """Load mask regions and shift corrections from a JSON config file."""
    if not mask_path or not os.path.exists(mask_path):
        return {}, None
    with open(mask_path, "r") as f:
        data = json.load(f)
    size = (data.get("imageWidth", 0), data.get("imageHeight", 0))
    return data, size if size != (0, 0) else None


def apply_mask(arr, masks, image_size):
    """
    Zero out masked regions in an array so they don't affect comparison.
    masks: list of {x, y, width, height} in image coordinates.
    image_size: (img_width, img_height) — the size the masks were drawn on.
    Returns masked copy.
    """
    if not masks:
        return arr
    masked = arr.copy()
    h, w = masked.shape[:2]
    # Scale mask coords if image was resized
    sx = w / image_size[0] if image_size[0] else 1
    sy = h / image_size[1] if image_size[1] else 1
    for m in masks:
        x1 = int(m["x"] * sx)
        y1 = int(m["y"] * sy)
        x2 = int((m["x"] + m["width"]) * sx)
        y2 = int((m["y"] + m["height"]) * sy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        masked[y1:y2, x1:x2] = 0
    return masked


def apply_shifts(new_arr, shifts, image_size):
    """
    Correct known pixel offsets in regions of the new image.
    Each shift: {x, y, width, height, shiftX, shiftY}
    The region in the NEW image is shifted by (-shiftX, -shiftY) to realign with OLD.
    """
    if not shifts:
        return new_arr
    corrected = new_arr.copy()
    h, w = corrected.shape[:2]
    sx = w / image_size[0] if image_size[0] else 1
    sy = h / image_size[1] if image_size[1] else 1

    for s in shifts:
        # Region bounds (where the content SHOULD be, i.e. old position)
        dst_x1 = max(0, int(s["x"] * sx))
        dst_y1 = max(0, int(s["y"] * sy))
        dst_x2 = min(w, int((s["x"] + s["width"]) * sx))
        dst_y2 = min(h, int((s["y"] + s["height"]) * sy))

        # Source in the new image (where the content actually IS)
        shift_x = int(s.get("shiftX", 0) * sx)
        shift_y = int(s.get("shiftY", 0) * sy)
        src_x1 = dst_x1 + shift_x
        src_y1 = dst_y1 + shift_y
        src_x2 = dst_x2 + shift_x
        src_y2 = dst_y2 + shift_y

        # Clamp source to image bounds
        src_x1_c = max(0, src_x1)
        src_y1_c = max(0, src_y1)
        src_x2_c = min(w, src_x2)
        src_y2_c = min(h, src_y2)

        # Corresponding destination
        d_x1 = dst_x1 + (src_x1_c - src_x1)
        d_y1 = dst_y1 + (src_y1_c - src_y1)
        d_x2 = d_x1 + (src_x2_c - src_x1_c)
        d_y2 = d_y1 + (src_y2_c - src_y1_c)

        if d_x2 > d_x1 and d_y2 > d_y1:
            corrected[d_y1:d_y2, d_x1:d_x2] = new_arr[src_y1_c:src_y2_c, src_x1_c:src_x2_c]

    return corrected


def create_mask_overlay(img, masks, image_size):
    """Draw semi-transparent mask regions on an image for visualization."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    w, h = img.size
    sx = w / image_size[0] if image_size[0] else 1
    sy = h / image_size[1] if image_size[1] else 1
    for m in masks:
        x1 = int(m["x"] * sx)
        y1 = int(m["y"] * sy)
        x2 = int((m["x"] + m["width"]) * sx)
        y2 = int((m["y"] + m["height"]) * sy)
        draw.rectangle([x1, y1, x2, y2], fill=(100, 100, 255, 80), outline=(100, 100, 255, 200), width=2)
    return overlay.convert("RGB")


# ---------------------------------------------------------------------------
# Color normalization (color-invariant comparison)
# ---------------------------------------------------------------------------

def compute_color_swap_score(old_arr, new_arr, n_colors=16, masks=None, image_size=None):
    """
    Compute how well the spatial color distribution matches between two images.
    Returns a score 0.0-1.0 where 1.0 means identical distribution (just colors swapped).

    Downscales images first to smooth out gradients, then runs k-means ONCE on the
    combined pixels of both images so that clustering is consistent. Uses Hungarian
    algorithm to find the best 1-to-1 color mapping based on spatial overlap, then
    measures what fraction of pixels have matching mapped labels.
    """
    h, w = old_arr.shape[:2]

    # Downscale to smooth out gradients and speed up clustering.
    SCALE = max(1, max(h, w) // 200)  # target ~200px on longest side
    small_h, small_w = max(1, h // SCALE), max(1, w // SCALE)
    old_small = np.array(Image.fromarray(old_arr).resize((small_w, small_h), Image.LANCZOS))
    new_small = np.array(Image.fromarray(new_arr).resize((small_w, small_h), Image.LANCZOS))

    # Build valid-pixel mask on the small image
    valid = np.ones((small_h, small_w), dtype=bool)
    if masks and image_size:
        sx = small_w / image_size[0] if image_size[0] else 1
        sy = small_h / image_size[1] if image_size[1] else 1
        for m in masks:
            x1 = max(0, int(m["x"] * sx))
            y1 = max(0, int(m["y"] * sy))
            x2 = min(small_w, int((m["x"] + m["width"]) * sx))
            y2 = min(small_h, int((m["y"] + m["height"]) * sy))
            valid[y1:y2, x1:x2] = False

    old_pixels = old_small[valid].reshape(-1, 3).astype(np.float32)
    new_pixels = new_small[valid].reshape(-1, 3).astype(np.float32)

    if len(old_pixels) == 0 or len(new_pixels) == 0:
        return 0.0

    # Run k-means ONCE on combined pixels from both images.
    # This ensures consistent cluster centers — identical images get identical
    # labels, and color-swapped images get stable, comparable clusters.
    combined = np.vstack([old_pixels, new_pixels])
    k = min(n_colors, len(combined))
    if k < 2:
        return 1.0  # trivial case

    centers, _ = kmeans2(combined, k, minit='++', iter=30)

    # Assign both images to the same centers
    old_labels_valid, _ = vq(old_pixels, centers)
    new_labels_valid, _ = vq(new_pixels, centers)
    old_labels_valid = old_labels_valid.astype(np.int32)
    new_labels_valid = new_labels_valid.astype(np.int32)

    # Build overlap matrix: overlap[i][j] = pixels where old==i AND new==j
    overlap = np.zeros((k, k), dtype=np.int64)
    np.add.at(overlap, (old_labels_valid, new_labels_valid), 1)

    # Hungarian algorithm (minimize cost = maximize overlap)
    row_ind, col_ind = linear_sum_assignment(-overlap)

    # Build mapping: new_cluster -> old_cluster
    new_to_old = {}
    for r, c in zip(row_ind, col_ind):
        new_to_old[c] = r

    # Remap new labels to old label space and measure overlap
    remap_table = np.arange(k, dtype=np.int32)  # default: identity
    for new_j, old_i in new_to_old.items():
        remap_table[new_j] = old_i
    remapped_new_labels = remap_table[new_labels_valid]

    # Match ratio = fraction of valid pixels where mapped labels agree
    total_valid = len(old_labels_valid)
    if total_valid == 0:
        return 0.0
    matching = np.sum(old_labels_valid == remapped_new_labels)
    return float(matching / total_valid)


# ---------------------------------------------------------------------------
# Core comparison functions
# ---------------------------------------------------------------------------

def load_and_prepare(path):
    """Load an image and convert to RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img), img


def compute_diff(old_arr, new_arr, masks=None, image_size=None):
    """Compute SSIM and per-pixel difference between two images."""
    old_cmp = apply_mask(old_arr, masks, image_size) if masks else old_arr
    new_cmp = apply_mask(new_arr, masks, image_size) if masks else new_arr

    old_gray = np.mean(old_cmp, axis=2).astype(np.uint8)
    new_gray = np.mean(new_cmp, axis=2).astype(np.uint8)
    score, ssim_map = ssim(old_gray, new_gray, full=True)

    pixel_diff = np.abs(old_cmp.astype(np.int16) - new_cmp.astype(np.int16)).astype(np.uint8)
    diff_mask = np.any(pixel_diff > 10, axis=2)

    return score, ssim_map, pixel_diff, diff_mask


def create_highlighted_diff(new_arr, diff_mask):
    """New image with bright red on changed pixels, dimmed unchanged pixels."""
    highlighted = new_arr.copy()
    # Dim the unchanged pixels so the red really pops
    highlighted[~diff_mask] = (highlighted[~diff_mask] * 0.3).astype(np.uint8)
    # Paint changed pixels solid bright red
    highlighted[diff_mask] = [255, 0, 0]

    # Add a bright outline around diff regions for extra visibility
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(diff_mask, iterations=3)
    outline = dilated & ~diff_mask
    highlighted[outline] = [255, 255, 0]  # yellow border around red regions

    return highlighted


def create_side_by_side(old_img, new_img, diff_img, score):
    """Side-by-side: Old | New | Diff."""
    w, h = old_img.size
    padding = 10
    label_height = 40
    total_w = w * 3 + padding * 4
    total_h = h + padding * 2 + label_height

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    labels = ["OLD (before refactor)", "NEW (after refactor)", f"DIFF (SSIM: {score:.4f})"]
    colors = [(200, 200, 200), (200, 200, 200), (255, 80, 80) if score < 0.99 else (80, 255, 80)]

    for i, (label, color) in enumerate(zip(labels, colors)):
        x_offset = padding + i * (w + padding)
        draw.text((x_offset + 10, 8), label, fill=color)
        canvas.paste(old_img if i == 0 else (new_img if i == 1 else diff_img),
                     (x_offset, label_height))
    return canvas


def create_amplified_diff(pixel_diff):
    """Amplify pixel differences so subtle changes pop."""
    return np.clip(pixel_diff.astype(np.float32) * 5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Single-level comparison (can run in a subprocess)
# ---------------------------------------------------------------------------

def run_comparison(old_path, new_path, output_dir, threshold=0.995, save_pass=False,
                   masks=None, mask_image_size=None, shifts=None, color_normalize_k=0):
    """
    Compare one pair. Returns a result dict.
    Only generates diff images for FAILed levels (unless save_pass=True).
    """
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(old_path))[0]

    try:
        old_arr, old_img = load_and_prepare(old_path)
        new_arr, new_img = load_and_prepare(new_path)
    except Exception as e:
        return {
            "level": basename,
            "old_path": old_path,
            "new_path": new_path,
            "error": str(e),
            "score": 0,
            "passed": False,
            "changed_pixels": 0,
            "change_pct": 0,
            "color_swapped": False,
            "norm_score": None,
            "images": {},
        }

    size_warning = None
    if old_arr.shape != new_arr.shape:
        size_warning = f"Resized: old {old_arr.shape} -> new {new_arr.shape}"
        new_img = new_img.resize(old_img.size, Image.LANCZOS)
        new_arr = np.array(new_img)

    # Use actual image size for mask/shift scaling if not provided
    img_size = mask_image_size or (old_img.size[0], old_img.size[1])

    # Apply shift correction to the new image before comparing
    if shifts:
        new_arr = apply_shifts(new_arr, shifts, img_size)
        new_img = Image.fromarray(new_arr)

    # Color-swap detection: measure spatial distribution overlap
    color_swapped = False
    norm_score = None
    if color_normalize_k and color_normalize_k > 1:
        norm_score = compute_color_swap_score(old_arr, new_arr, n_colors=color_normalize_k,
                                               masks=masks, image_size=img_size)

    score, ssim_map, pixel_diff, diff_mask = compute_diff(old_arr, new_arr, masks, img_size)
    changed_pixels = int(np.sum(diff_mask))
    total_pixels = int(diff_mask.size)
    change_pct = (changed_pixels / total_pixels) * 100
    passed = score >= threshold

    # If normal SSIM fails but color distribution matches (>90%), it's a color swap
    if color_normalize_k and color_normalize_k > 1 and not passed:
        if norm_score is not None and norm_score >= 0.90:
            passed = True
            color_swapped = True

    images = {}

    if not passed or save_pass:
        highlighted = create_highlighted_diff(new_arr, diff_mask)
        highlighted_img = Image.fromarray(highlighted)

        # Overlay mask regions on highlighted if masks exist
        if masks:
            highlighted_img = create_mask_overlay(highlighted_img, masks, img_size)

        p = os.path.join(output_dir, f"{basename}_highlighted.png")
        highlighted_img.save(p)
        images["highlighted"] = p

        amplified = create_amplified_diff(pixel_diff)
        amplified_img = Image.fromarray(amplified)
        p = os.path.join(output_dir, f"{basename}_amplified.png")
        amplified_img.save(p)
        images["amplified"] = p

        side = create_side_by_side(old_img, new_img, highlighted_img, score)
        p = os.path.join(output_dir, f"{basename}_sidebyside.png")
        side.save(p)
        images["sidebyside"] = p

        ssim_visual = (ssim_map * 255).astype(np.uint8)
        ssim_img = Image.fromarray(ssim_visual, mode="L")
        p = os.path.join(output_dir, f"{basename}_ssim.png")
        ssim_img.save(p)
        images["ssim"] = p

    thumb_size = (200, 300)
    for tag, img in [("thumb_old", old_img), ("thumb_new", new_img)]:
        thumb = img.copy()
        thumb.thumbnail(thumb_size, Image.LANCZOS)
        p = os.path.join(output_dir, f"{basename}_{tag}.png")
        thumb.save(p)
        images[tag] = p

    return {
        "level": basename,
        "old_path": old_path,
        "new_path": new_path,
        "score": float(score),
        "passed": passed,
        "changed_pixels": changed_pixels,
        "change_pct": round(change_pct, 3),
        "size_warning": size_warning,
        "color_swapped": color_swapped,
        "norm_score": float(norm_score) if norm_score is not None else None,
        "error": None,
        "images": images,
    }


# Worker function for multiprocessing
def _worker(args):
    return run_comparison(*args)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def extract_level_key(filename):
    """
    Extract the level identifier from a filename.
    e.g. 'Level30_VeryHard_30_9C_TH_VERYHARD.png' -> 'Level30'
         'Level30_Easy_Level.png' -> 'Level30'
         'level_190.png' -> 'level_190'

    Matches patterns like:
      - Level30  (word + digits)
      - level_190 (word + underscore + digits)
    """
    import re
    base = os.path.splitext(filename)[0]

    # Try: "Level" followed by digits (case insensitive)
    m = re.match(r'(Level\d+)', base, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Try: "level_" followed by digits
    m = re.match(r'(level_\d+)', base, re.IGNORECASE)
    if m:
        return m.group(1).lower()

    # Fallback: use the full filename (original behavior)
    return base.lower()


def find_pairs(old_dir, new_dir):
    """
    Find matching PNG files across both directories.
    Matches by level key (e.g. 'Level30') so filenames don't need to be identical.
    """
    old_files = [f for f in os.listdir(old_dir) if f.lower().endswith(".png")]
    new_files = [f for f in os.listdir(new_dir) if f.lower().endswith(".png")]

    # Build key -> filename maps
    old_by_key = {}
    for f in old_files:
        key = extract_level_key(f)
        old_by_key[key] = f

    new_by_key = {}
    for f in new_files:
        key = extract_level_key(f)
        new_by_key[key] = f

    # Match by key
    matched_keys = sorted(set(old_by_key.keys()) & set(new_by_key.keys()))
    only_old_keys = sorted(set(old_by_key.keys()) - set(new_by_key.keys()))
    only_new_keys = sorted(set(new_by_key.keys()) - set(old_by_key.keys()))

    pairs = [
        (os.path.join(old_dir, old_by_key[k]), os.path.join(new_dir, new_by_key[k]))
        for k in matched_keys
    ]
    only_old = [old_by_key[k] for k in only_old_keys]
    only_new = [new_by_key[k] for k in only_new_keys]

    return pairs, only_old, only_new


def run_batch(old_dir, new_dir, output_dir, threshold=0.995, workers=4, save_pass=False,
              masks=None, mask_image_size=None, shifts=None, color_normalize_k=0):
    """Process all matched pairs, return list of results."""
    pairs, only_old, only_new = find_pairs(old_dir, new_dir)

    if not pairs:
        print("ERROR: No matching PNG files found in both directories.")
        print(f"  Old dir ({old_dir}): {len(only_old)} files")
        print(f"  New dir ({new_dir}): {len(only_new)} files")
        sys.exit(1)

    diff_dir = os.path.join(output_dir, "diffs")
    os.makedirs(diff_dir, exist_ok=True)

    print(f"\nProcessing {len(pairs)} level pairs...")
    if masks:
        print(f"  Using {len(masks)} mask region(s)")
    if shifts:
        print(f"  Using {len(shifts)} shift correction(s)")
    if color_normalize_k:
        print(f"  Color normalization: ON (K={color_normalize_k})")
    if only_old:
        print(f"  Warning: {len(only_old)} files only in old dir (missing from new)")
    if only_new:
        print(f"  Warning: {len(only_new)} files only in new dir (missing from old)")
    print()

    tasks = [
        (old_p, new_p, diff_dir, threshold, save_pass, masks, mask_image_size, shifts, color_normalize_k)
        for old_p, new_p in pairs
    ]
    results = []

    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in tasks}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            status = "SWAP" if result.get("color_swapped") else ("PASS" if result["passed"] else "FAIL")
            print(f"  [{done_count:>4}/{len(pairs)}] {status}  SSIM={result['score']:.4f}  {result['level']}")
            results.append(result)

    elapsed = time.time() - start
    results.sort(key=lambda r: r["score"])

    print(f"\nDone in {elapsed:.1f}s")

    return results, only_old, only_new


# ---------------------------------------------------------------------------
# HTML report (same as before)
# ---------------------------------------------------------------------------

def generate_html_report(results, only_old, only_new, output_dir, threshold):
    """Generate an interactive HTML report."""
    total = len(results)
    failed = [r for r in results if not r["passed"]]
    swapped = [r for r in results if r["passed"] and r.get("color_swapped")]
    passed = [r for r in results if r["passed"] and not r.get("color_swapped")]

    report_path = os.path.join(output_dir, "report.html")

    def img_tag(path, alt="", css_class="thumb"):
        if not path or not os.path.exists(path):
            return f'<span class="no-img">{alt}</span>'
        rel = os.path.relpath(path, output_dir)
        return f'<img src="{rel}" alt="{alt}" class="{css_class}" loading="lazy">'

    def score_bar(score, threshold):
        pct = score * 100
        color = "#4caf50" if score >= threshold else ("#ff9800" if score >= threshold * 0.98 else "#f44336")
        return f'''<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;background:{color}"></div></div>'''

    rows_failed = ""
    for r in failed:
        imgs = r["images"]
        rows_failed += f'''
        <div class="level-card fail" onclick="this.classList.toggle('expanded')">
            <div class="card-header">
                <span class="status-badge fail">FAIL</span>
                <span class="level-name">{r['level']}</span>
                <span class="score">SSIM: {r['score']:.6f}</span>
                <span class="change-pct">{r['change_pct']:.2f}% changed</span>
                {score_bar(r['score'], threshold)}
            </div>
            <div class="card-details">
                <div class="thumbs">
                    <div class="thumb-col">
                        <div class="thumb-label">Old</div>
                        {img_tag(imgs.get('thumb_old'), 'old')}
                    </div>
                    <div class="thumb-col">
                        <div class="thumb-label">New</div>
                        {img_tag(imgs.get('thumb_new'), 'new')}
                    </div>
                </div>
                <div class="diff-images">
                    <div class="diff-section">
                        <div class="diff-label">Side by Side</div>
                        {img_tag(imgs.get('sidebyside'), 'side-by-side', 'full-img')}
                    </div>
                    <div class="diff-section">
                        <div class="diff-label">Highlighted Changes</div>
                        {img_tag(imgs.get('highlighted'), 'highlighted', 'full-img')}
                    </div>
                    <div class="diff-section">
                        <div class="diff-label">Amplified Diff</div>
                        {img_tag(imgs.get('amplified'), 'amplified', 'full-img')}
                    </div>
                </div>
                {f'<div class="warning">{r["size_warning"]}</div>' if r.get("size_warning") else ""}
                {f'<div class="error">Error: {r["error"]}</div>' if r.get("error") else ""}
            </div>
        </div>'''

    rows_swapped = ""
    for r in swapped:
        imgs = r["images"]
        norm_info = f"  Distribution match: {r['norm_score']:.1%}" if r.get('norm_score') is not None else ""
        rows_swapped += f'''
        <div class="level-card swap" onclick="this.classList.toggle('expanded')">
            <div class="card-header">
                <span class="status-badge swap">SWAP</span>
                <span class="level-name">{r['level']}</span>
                <span class="score">SSIM: {r['score']:.6f}{norm_info}</span>
                <span class="change-pct" style="color:#ff9800">{r['change_pct']:.2f}% changed</span>
                {score_bar(r['score'], threshold)}
            </div>
            <div class="card-details">
                <div style="color:#ff9800;font-size:13px;margin-bottom:12px;">Colors changed but distribution matches (color-invariant check passed)</div>
                <div class="thumbs">
                    <div class="thumb-col">
                        <div class="thumb-label">Old</div>
                        {img_tag(imgs.get('thumb_old'), 'old')}
                    </div>
                    <div class="thumb-col">
                        <div class="thumb-label">New</div>
                        {img_tag(imgs.get('thumb_new'), 'new')}
                    </div>
                </div>
            </div>
        </div>'''

    rows_passed = ""
    for r in passed:
        imgs = r["images"]
        rows_passed += f'''
        <div class="level-card pass">
            <div class="card-header">
                <span class="status-badge pass">PASS</span>
                <span class="level-name">{r['level']}</span>
                <span class="score">SSIM: {r['score']:.6f}</span>
                {score_bar(r['score'], threshold)}
            </div>
        </div>'''

    missing_html = ""
    if only_old or only_new:
        missing_html = '<div class="section"><h2>Missing Files</h2>'
        if only_old:
            missing_html += f'<details><summary>{len(only_old)} files only in old folder</summary><ul>'
            for f in only_old:
                missing_html += f'<li>{f}</li>'
            missing_html += '</ul></details>'
        if only_new:
            missing_html += f'<details><summary>{len(only_new)} files only in new folder</summary><ul>'
            for f in only_new:
                missing_html += f'<li>{f}</li>'
            missing_html += '</ul></details>'
        missing_html += '</div>'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Regression Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }}
    .header {{ text-align: center; padding: 30px 0; border-bottom: 2px solid #333; margin-bottom: 30px; }}
    .header h1 {{ font-size: 28px; margin-bottom: 10px; }}
    .header .subtitle {{ color: #888; font-size: 14px; }}
    .summary {{ display: flex; gap: 20px; justify-content: center; margin-bottom: 30px; flex-wrap: wrap; }}
    .summary-card {{ background: #16213e; border-radius: 12px; padding: 20px 30px; text-align: center; min-width: 140px; }}
    .summary-card .number {{ font-size: 36px; font-weight: 700; }}
    .summary-card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
    .summary-card.fail .number {{ color: #f44336; }}
    .summary-card.pass .number {{ color: #4caf50; }}
    .summary-card.warn .number {{ color: #ff9800; }}
    .summary-card.total .number {{ color: #64b5f6; }}
    .section {{ margin-bottom: 30px; }}
    .section h2 {{ font-size: 20px; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 1px solid #333; }}
    .filter-bar {{ display: flex; gap: 10px; margin-bottom: 20px; align-items: center; }}
    .filter-bar input {{ background: #16213e; border: 1px solid #333; color: #e0e0e0; padding: 8px 14px; border-radius: 8px; font-size: 14px; width: 300px; }}
    .filter-bar .filter-btn {{ background: #16213e; border: 1px solid #333; color: #e0e0e0; padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; }}
    .filter-bar .filter-btn.active {{ background: #0f3460; border-color: #64b5f6; color: #64b5f6; }}
    .level-card {{ background: #16213e; border-radius: 10px; margin-bottom: 8px; overflow: hidden; }}
    .level-card.fail {{ border-left: 4px solid #f44336; }}
    .level-card.pass {{ border-left: 4px solid #4caf50; }}
    .level-card.swap {{ border-left: 4px solid #ff9800; }}
    .level-card.fail .card-header {{ cursor: pointer; }}
    .level-card.fail .card-header:hover {{ background: #1a2744; }}
    .level-card.swap .card-header {{ cursor: pointer; }}
    .level-card.swap .card-header:hover {{ background: #1a2744; }}
    .card-header {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px; flex-wrap: wrap; }}
    .status-badge {{ font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px; }}
    .status-badge.fail {{ background: #f4433622; color: #f44336; }}
    .status-badge.pass {{ background: #4caf5022; color: #4caf50; }}
    .status-badge.swap {{ background: #ff980022; color: #ff9800; }}
    .summary-card.swap .number {{ color: #ff9800; }}
    .level-name {{ font-weight: 600; min-width: 120px; }}
    .score {{ color: #888; font-size: 13px; font-family: monospace; }}
    .change-pct {{ color: #ff9800; font-size: 13px; }}
    .score-bar-bg {{ flex: 1; min-width: 100px; max-width: 200px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }}
    .score-bar-fill {{ height: 100%; border-radius: 3px; }}
    .card-details {{ display: none; padding: 16px; border-top: 1px solid #333; }}
    .level-card.expanded .card-details {{ display: block; }}
    .thumbs {{ display: flex; gap: 20px; margin-bottom: 16px; }}
    .thumb-col {{ text-align: center; }}
    .thumb-label {{ font-size: 12px; color: #888; margin-bottom: 6px; }}
    .thumb {{ max-height: 300px; border-radius: 6px; border: 1px solid #333; }}
    .diff-images {{ display: flex; flex-direction: column; gap: 16px; }}
    .diff-label {{ font-size: 13px; color: #64b5f6; margin-bottom: 6px; font-weight: 600; }}
    .full-img {{ max-width: 100%; border-radius: 6px; border: 1px solid #333; cursor: pointer; }}
    .full-img:hover {{ border-color: #64b5f6; }}
    .warning {{ color: #ff9800; font-size: 13px; margin-top: 8px; }}
    .error {{ color: #f44336; font-size: 13px; margin-top: 8px; }}
    .no-img {{ color: #666; font-style: italic; }}
    details {{ margin: 8px 0; }}
    details summary {{ cursor: pointer; color: #ff9800; }}
    details ul {{ padding-left: 24px; margin-top: 8px; }}
    details li {{ margin: 2px 0; color: #888; font-size: 13px; }}
    .lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.9); z-index: 1000; align-items: center; justify-content: center; cursor: zoom-out; }}
    .lightbox.active {{ display: flex; }}
    .lightbox img {{ max-width: 95vw; max-height: 95vh; border-radius: 8px; }}
</style>
</head>
<body>
<div class="header">
    <h1>Visual Regression Report</h1>
    <div class="subtitle">Threshold: {threshold} &middot; Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
</div>
<div class="summary">
    <div class="summary-card total"><div class="number">{total}</div><div class="label">Total Levels</div></div>
    <div class="summary-card fail"><div class="number">{len(failed)}</div><div class="label">Failed</div></div>
    <div class="summary-card swap"><div class="number">{len(swapped)}</div><div class="label">Color Swaps</div></div>
    <div class="summary-card pass"><div class="number">{len(passed)}</div><div class="label">Passed</div></div>
    <div class="summary-card warn"><div class="number">{len(only_old) + len(only_new)}</div><div class="label">Missing</div></div>
</div>
<div class="filter-bar">
    <input type="text" id="search" placeholder="Search levels..." oninput="filterLevels()">
    <button class="filter-btn active" onclick="setFilter('all', this)">All</button>
    <button class="filter-btn" onclick="setFilter('fail', this)">Failures only</button>
    <button class="filter-btn" onclick="setFilter('swap', this)">Color swaps</button>
    <button class="filter-btn" onclick="setFilter('pass', this)">Passed only</button>
</div>
{missing_html}
<div class="section" id="results-section">
    <h2>Failed ({len(failed)})</h2>
    <div id="failed-list">{rows_failed}</div>
    <h2 style="margin-top:30px">Color Swaps ({len(swapped)})</h2>
    <div id="swap-list">{rows_swapped}</div>
    <h2 style="margin-top:30px">Passed ({len(passed)})</h2>
    <div id="passed-list">{rows_passed}</div>
</div>
<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
    <img id="lightbox-img" src="" alt="Full size">
</div>
<script>
document.querySelectorAll('.full-img').forEach(img => {{
    img.addEventListener('click', e => {{
        e.stopPropagation();
        document.getElementById('lightbox-img').src = img.src;
        document.getElementById('lightbox').classList.add('active');
    }});
}});
let currentFilter = 'all';
function setFilter(filter, btn) {{
    currentFilter = filter;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filterLevels();
}}
function filterLevels() {{
    const query = document.getElementById('search').value.toLowerCase();
    document.querySelectorAll('.level-card').forEach(card => {{
        const name = card.querySelector('.level-name')?.textContent.toLowerCase() || '';
        const matchesSearch = !query || name.includes(query);
        const isFail = card.classList.contains('fail');
        const isSwap = card.classList.contains('swap');
        const isPass = card.classList.contains('pass');
        const matchesFilter = currentFilter === 'all'
            || (currentFilter === 'fail' && isFail)
            || (currentFilter === 'swap' && isSwap)
            || (currentFilter === 'pass' && isPass);
        card.style.display = (matchesSearch && matchesFilter) ? '' : 'none';
    }});
}}
</script>
</body>
</html>'''

    with open(report_path, "w") as f:
        f.write(html)
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visual regression diff tool for game level screenshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 visual_diff.py old/ new/
  python3 visual_diff.py old/ new/ --mask mask_config.json
  python3 visual_diff.py --single old/level_190.png new/level_190.png
        """,
    )
    parser.add_argument("old", help="Old screenshots directory (or old image if --single)")
    parser.add_argument("new", help="New screenshots directory (or new image if --single)")
    parser.add_argument("--output-dir", "-o", default="regression_report",
                        help="Output directory (default: regression_report/)")
    parser.add_argument("--threshold", "-t", type=float, default=0.995,
                        help="SSIM threshold for PASS/FAIL (default: 0.995)")
    parser.add_argument("--single", action="store_true",
                        help="Compare a single pair instead of batch")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--save-pass", action="store_true",
                        help="Also save diff images for passed levels")
    parser.add_argument("--color-normalize", type=int, default=0, metavar="K",
                        help="Enable color-invariant comparison with K color clusters "
                             "(e.g. --color-normalize 16). Detects color swaps without "
                             "false positives. 0 = disabled (default).")
    parser.add_argument("--mask", "-m", default=None,
                        help="Path to mask config JSON (from the web UI)")
    args = parser.parse_args()

    masks = None
    shifts = None
    mask_image_size = None
    if args.mask:
        config, mask_image_size = load_mask_config(args.mask)
        masks = config.get("masks") or None
        shifts = config.get("shifts") or None

    if args.single:
        if not os.path.isfile(args.old):
            print(f"Error: File not found: {args.old}")
            sys.exit(1)
        if not os.path.isfile(args.new):
            print(f"Error: File not found: {args.new}")
            sys.exit(1)
        result = run_comparison(args.old, args.new, args.output_dir, args.threshold,
                                save_pass=True, masks=masks, mask_image_size=mask_image_size,
                                shifts=shifts, color_normalize_k=args.color_normalize)
        status = "SWAP" if result.get("color_swapped") else ("PASS" if result["passed"] else "FAIL")
        extra = ""
        if result.get("norm_score") is not None:
            extra = f"  Norm-SSIM: {result['norm_score']:.6f}"
        print(f"\n{status}  SSIM: {result['score']:.6f}{extra}  Changed: {result['change_pct']:.2f}%")
    else:
        if not os.path.isdir(args.old):
            print(f"Error: Not a directory: {args.old}")
            sys.exit(1)
        if not os.path.isdir(args.new):
            print(f"Error: Not a directory: {args.new}")
            sys.exit(1)

        results, only_old, only_new = run_batch(
            args.old, args.new, args.output_dir, args.threshold, args.workers, args.save_pass,
            masks, mask_image_size, shifts, args.color_normalize
        )

        failed = [r for r in results if not r["passed"]]
        swapped = [r for r in results if r["passed"] and r.get("color_swapped")]
        passed = [r for r in results if r["passed"] and not r.get("color_swapped")]

        print(f"\n{'='*60}")
        swap_info = f"  |  Color Swaps: {len(swapped)}" if swapped else ""
        print(f"  Total: {len(results)}  |  Passed: {len(passed)}{swap_info}  |  Failed: {len(failed)}")
        if only_old:
            print(f"  Missing from new: {len(only_old)}")
        if only_new:
            print(f"  Missing from old: {len(only_new)}")
        print(f"{'='*60}")

        report_path = generate_html_report(results, only_old, only_new, args.output_dir, args.threshold)
        print(f"\n  Report: {report_path}")
        print(f"  Diff images: {os.path.join(args.output_dir, 'diffs')}/")


if __name__ == "__main__":
    main()
