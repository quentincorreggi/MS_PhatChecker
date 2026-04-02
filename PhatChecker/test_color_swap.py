#!/usr/bin/env python3
"""
Test script for color swap detection.
Creates synthetic game-like screenshots and tests the algorithm.
"""
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2, vq
from scipy.optimize import linear_sum_assignment


def _kmeans_quantize(pixels, n_colors):
    if len(pixels) < n_colors:
        k = max(1, len(pixels))
    else:
        k = n_colors
    if k < 2:
        center = pixels[:1] if len(pixels) > 0 else np.zeros((1, 3), dtype=np.float32)
        return np.zeros(len(pixels), dtype=np.int32), center
    centers, labels = kmeans2(pixels, k, minit='++', iter=30)
    return labels.astype(np.int32), centers


def compute_color_swap_score(old_arr, new_arr, n_colors=16, masks=None, image_size=None):
    h, w = old_arr.shape[:2]

    # Downscale to smooth gradients
    SCALE = max(1, max(h, w) // 200)
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

    old_labels_valid, old_centers = _kmeans_quantize(old_pixels, n_colors)
    new_labels_valid, new_centers = _kmeans_quantize(new_pixels, n_colors)

    k_old = len(old_centers)
    k_new = len(new_centers)

    overlap = np.zeros((k_old, k_new), dtype=np.int64)
    np.add.at(overlap, (old_labels_valid, new_labels_valid), 1)

    row_ind, col_ind = linear_sum_assignment(-overlap)

    new_to_old = {}
    for r, c in zip(row_ind, col_ind):
        new_to_old[c] = r
    matched_new = set(col_ind)
    for j in range(k_new):
        if j not in matched_new:
            dists = np.linalg.norm(old_centers - new_centers[j], axis=1)
            new_to_old[j] = int(np.argmin(dists))

    remap_table = np.full(k_new, -1, dtype=np.int32)
    for new_j, old_i in new_to_old.items():
        remap_table[new_j] = old_i
    remapped_new_labels = remap_table[new_labels_valid]

    total_valid = len(old_labels_valid)
    if total_valid == 0:
        return 0.0
    matching = np.sum(old_labels_valid == remapped_new_labels)
    return float(matching / total_valid)


def create_game_screenshot(bg_color_top, bg_color_bot, block_colors, w=540, h=960):
    """
    Create a synthetic game screenshot with:
    - Gradient background (bg_color_top to bg_color_bot)
    - A gray puzzle grid area in the center
    - Colored blocks placed on the grid
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for y in range(h):
        t = y / h
        for c in range(3):
            img[y, :, c] = int(bg_color_top[c] * (1 - t) + bg_color_bot[c] * t)

    # Gray puzzle grid area (center of image)
    grid_y1, grid_y2 = 300, 700
    grid_x1, grid_x2 = 70, 470
    img[grid_y1:grid_y2, grid_x1:grid_x2] = [140, 150, 160]

    # Place colored blocks on a 5x5 grid within the puzzle area
    block_w = 70
    block_h = 70
    gap = 10
    layout = [
        # row 0: block_colors[0], empty, block_colors[1], empty, block_colors[2]
        [0, -1, 1, -1, 2],
        # row 1: block_colors[3], block_colors[4], block_colors[0], block_colors[1], block_colors[3]
        [3, 4, 0, 1, 3],
        # row 2: empty, block_colors[2], block_colors[3], block_colors[4], empty
        [-1, 2, 3, 4, -1],
        # row 3: block_colors[1], block_colors[0], empty, block_colors[2], block_colors[4]
        [1, 0, -1, 2, 4],
        # row 4: block_colors[0], block_colors[3], block_colors[2], block_colors[1], block_colors[0]
        [0, 3, 2, 1, 0],
    ]

    for row_i, row in enumerate(layout):
        for col_i, color_idx in enumerate(row):
            if color_idx < 0:
                continue
            x = grid_x1 + col_i * (block_w + gap)
            y = grid_y1 + 20 + row_i * (block_h + gap)
            color = block_colors[color_idx]
            # Solid block with slight 3D effect (lighter top, darker bottom)
            img[y:y+block_h, x:x+block_w] = color
            img[y:y+3, x:x+block_w] = [min(255, c + 30) for c in color]  # highlight
            img[y+block_h-3:y+block_h, x:x+block_w] = [max(0, c - 30) for c in color]  # shadow

    # Bottom area: blocks in a row (like the candy tray)
    tray_y = 770
    for i, color_idx in enumerate([0, 1, 2, 3, 4, 0, 1]):
        x = 30 + i * (block_w + 5)
        color = block_colors[color_idx]
        img[tray_y:tray_y+50, x:x+block_w] = color

    return img


def main():
    print("=" * 60)
    print("Color Swap Detection Test")
    print("=" * 60)

    # --- Test 1: Color swap (should detect as swap) ---
    print("\n--- Test 1: Color Swap (same layout, different colors) ---")
    old_colors = [
        [200, 50, 50],    # red
        [50, 50, 200],    # blue
        [50, 200, 50],    # green
        [200, 200, 50],   # yellow
        [200, 50, 200],   # purple
    ]
    new_colors = [
        [50, 200, 200],   # cyan (was red)
        [200, 100, 50],   # orange (was blue)
        [200, 50, 200],   # purple (was green)
        [100, 200, 100],  # light green (was yellow)
        [50, 50, 200],    # blue (was purple)
    ]

    old_img = create_game_screenshot(
        bg_color_top=[80, 180, 80], bg_color_bot=[40, 120, 40],  # green gradient
        block_colors=old_colors,
    )
    new_img = create_game_screenshot(
        bg_color_top=[180, 80, 80], bg_color_bot=[120, 40, 40],  # red/pink gradient
        block_colors=new_colors,
    )

    # Save for visual inspection
    Image.fromarray(old_img).save("/tmp/test_old.png")
    Image.fromarray(new_img).save("/tmp/test_new.png")
    print(f"  Saved test images to /tmp/test_old.png, /tmp/test_new.png")

    for k in [8, 12, 16, 24]:
        score = compute_color_swap_score(old_img, new_img, n_colors=k)
        status = "SWAP" if score >= 0.90 else "FAIL"
        print(f"  K={k:2d}  match_ratio={score:.4f}  -> {status}")

    # --- Test 2: Identical images (should be ~1.0) ---
    print("\n--- Test 2: Identical Images ---")
    for k in [8, 16]:
        score = compute_color_swap_score(old_img, old_img, n_colors=k)
        print(f"  K={k:2d}  match_ratio={score:.4f}")

    # --- Test 3: Real regression - block moved (should be low) ---
    print("\n--- Test 3: Real Regression (block position changed) ---")
    # Same colors but different layout
    regression_colors = old_colors  # same colors
    regression_img = create_game_screenshot(
        bg_color_top=[80, 180, 80], bg_color_bot=[40, 120, 40],
        block_colors=regression_colors,
    )
    # Manually move a block to a different position
    regression_img[400:470, 70:140] = [200, 50, 50]  # add red block where there was none
    regression_img[320:390, 70:140] = [140, 150, 160]  # remove block, replace with grid

    for k in [8, 16]:
        score = compute_color_swap_score(old_img, regression_img, n_colors=k)
        status = "SWAP" if score >= 0.90 else "FAIL"
        print(f"  K={k:2d}  match_ratio={score:.4f}  -> {status}")

    # --- Test 4: Color swap WITH gradient background (the real scenario) ---
    print("\n--- Test 4: Color Swap + Heavy Gradient Background ---")
    # Make the gradient more extreme to simulate the real game
    old_heavy = create_game_screenshot(
        bg_color_top=[100, 220, 100], bg_color_bot=[30, 80, 30],  # extreme green gradient
        block_colors=old_colors,
    )
    new_heavy = create_game_screenshot(
        bg_color_top=[220, 100, 130], bg_color_bot=[80, 30, 50],  # extreme pink gradient
        block_colors=new_colors,
    )

    for k in [8, 12, 16, 24]:
        score = compute_color_swap_score(old_heavy, new_heavy, n_colors=k)
        status = "SWAP" if score >= 0.90 else "FAIL"
        print(f"  K={k:2d}  match_ratio={score:.4f}  -> {status}")

    print("\n" + "=" * 60)
    print("Expected: Tests 1 & 4 should show SWAP, Test 3 should show FAIL")
    print("=" * 60)


if __name__ == "__main__":
    main()
