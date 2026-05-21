#!/usr/bin/env python3
"""
Color-agnostic pattern extraction and comparison for grid-based level screenshots.

Workflow:
  1. Load a grid config (the 7x7 top + 5x7 bottom layout, calibrated once).
  2. For each cell, sample the central region, mask out the background, and
     compute a robust hue+saturation summary.
  3. Empty cells (no ingredient) are detected by low coverage of saturated pixels
     and marked with the EMPTY_LABEL.
  4. Non-empty cells are clustered (k-means in HS-space) to assign each cell a
     short label ('A', 'B', 'C', ...). The labels are color-agnostic: swapping
     all blues for reds yields the same pattern.
  5. Two patterns are compared by finding the best bijection between their
     label sets (brute-force over k! permutations, k <= MAX_COLORS).
"""

from __future__ import annotations

import json
import os
import string
from dataclasses import dataclass, field, asdict
from itertools import permutations
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


EMPTY_LABEL = "."
LABEL_ALPHABET = string.ascii_uppercase  # A-Z, plenty for <= 8 colors

# Display palette for label grids in reports — picked to be visually distinct.
DISPLAY_PALETTE_RGB = [
    (244,  67,  54),  # red
    ( 33, 150, 243),  # blue
    ( 76, 175,  80),  # green
    (255, 152,   0),  # orange
    (156,  39, 176),  # purple
    (  0, 188, 212),  # cyan
    (255, 235,  59),  # yellow
    (233,  30,  99),  # pink
]


# ---------------------------------------------------------------------------
# Grid config
# ---------------------------------------------------------------------------

@dataclass
class GridSpec:
    name: str
    rows: int
    cols: int
    # Pixel coords on the calibration image
    x: int
    y: int
    width: int
    height: int


@dataclass
class GridConfig:
    image_width: int
    image_height: int
    grids: list[GridSpec]
    # Inner fraction of each cell used for sampling (avoids cell border bleed).
    cell_inset_pct: float = 0.20
    # Saturation threshold for "this pixel is part of an ingredient, not bg".
    sat_threshold: float = 0.25
    # If fewer than this fraction of cell pixels pass the sat threshold,
    # the cell is considered empty.
    empty_coverage_pct: float = 0.08
    # Min/max colors to try for auto-k clustering.
    min_k: int = 2
    max_k: int = 8
    # Hue gap (in degrees) for DBSCAN-on-the-hue-circle. Two cells whose hues
    # differ by less than this end up in the same cluster. ~30° works well
    # for distinct ingredient colors with gradients/shadows.
    hue_gap_deg: float = 30.0

    @classmethod
    def from_dict(cls, d: dict) -> "GridConfig":
        return cls(
            image_width=int(d.get("imageWidth", 0)),
            image_height=int(d.get("imageHeight", 0)),
            grids=[GridSpec(
                name=g["name"],
                rows=int(g["rows"]),
                cols=int(g["cols"]),
                x=int(g["x"]),
                y=int(g["y"]),
                width=int(g["width"]),
                height=int(g["height"]),
            ) for g in d.get("grids", [])],
            cell_inset_pct=float(d.get("cellInsetPct", 0.20)),
            sat_threshold=float(d.get("satThreshold", 0.25)),
            empty_coverage_pct=float(d.get("emptyCoveragePct", 0.08)),
            min_k=int(d.get("minK", 2)),
            max_k=int(d.get("maxK", 8)),
            hue_gap_deg=float(d.get("hueGapDeg", 30.0)),
        )

    def to_dict(self) -> dict:
        return {
            "imageWidth": self.image_width,
            "imageHeight": self.image_height,
            "grids": [asdict(g) for g in self.grids],
            "cellInsetPct": self.cell_inset_pct,
            "satThreshold": self.sat_threshold,
            "emptyCoveragePct": self.empty_coverage_pct,
            "minK": self.min_k,
            "maxK": self.max_k,
            "hueGapDeg": self.hue_gap_deg,
        }


def load_grid_config(path: str) -> Optional[GridConfig]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return GridConfig.from_dict(json.load(f))


def save_grid_config(path: str, config: GridConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Cell sampling
# ---------------------------------------------------------------------------

def _scale_factors(arr: np.ndarray, config: GridConfig) -> tuple[float, float]:
    """Scale grid coords from the calibration size to the actual image size."""
    h, w = arr.shape[:2]
    sx = w / config.image_width if config.image_width else 1.0
    sy = h / config.image_height if config.image_height else 1.0
    return sx, sy


def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Vectorized RGB[0..255] -> HSV[0..1]. Shape (...,3)."""
    rgb_f = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    cmax = np.max(rgb_f, axis=-1)
    cmin = np.min(rgb_f, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask = delta > 1e-8
    rc = np.where(cmax == r, ((g - b) / np.where(delta == 0, 1, delta)) % 6, 0)
    gc = np.where(cmax == g, ((b - r) / np.where(delta == 0, 1, delta)) + 2, 0)
    bc = np.where(cmax == b, ((r - g) / np.where(delta == 0, 1, delta)) + 4, 0)
    h = np.where(cmax == r, rc, np.where(cmax == g, gc, bc))
    h = (h / 6.0) % 1.0
    h = np.where(mask, h, 0.0)

    s = np.where(cmax > 1e-8, delta / np.where(cmax == 0, 1, cmax), 0.0)
    v = cmax
    return np.stack([h, s, v], axis=-1)


def _circular_mean_hue(hues: np.ndarray) -> float:
    """Mean of hues on the unit circle (hues in [0,1])."""
    if hues.size == 0:
        return 0.0
    angles = hues * 2.0 * np.pi
    x = float(np.mean(np.cos(angles)))
    y = float(np.mean(np.sin(angles)))
    mean = np.arctan2(y, x) / (2.0 * np.pi)
    if mean < 0:
        mean += 1.0
    return mean


@dataclass
class CellSample:
    grid: str
    row: int
    col: int
    bounds: tuple[int, int, int, int]  # x1,y1,x2,y2 of the sampled inner region
    empty: bool
    # Aggregate color of ingredient pixels (only valid when not empty).
    hue: float = 0.0
    sat: float = 0.0
    val: float = 0.0
    coverage: float = 0.0


def sample_cells(image_arr: np.ndarray, config: GridConfig) -> list[CellSample]:
    """Return one CellSample per grid cell across all grids."""
    sx, sy = _scale_factors(image_arr, config)
    h_img, w_img = image_arr.shape[:2]
    samples: list[CellSample] = []

    for grid in config.grids:
        gx = grid.x * sx
        gy = grid.y * sy
        gw = grid.width * sx
        gh = grid.height * sy
        cw = gw / grid.cols
        ch = gh / grid.rows
        inset = config.cell_inset_pct

        for r in range(grid.rows):
            for c in range(grid.cols):
                cx0 = gx + c * cw
                cy0 = gy + r * ch
                # Inner sampling rectangle
                ix1 = int(round(cx0 + cw * inset))
                iy1 = int(round(cy0 + ch * inset))
                ix2 = int(round(cx0 + cw * (1 - inset)))
                iy2 = int(round(cy0 + ch * (1 - inset)))
                ix1 = max(0, min(w_img - 1, ix1))
                iy1 = max(0, min(h_img - 1, iy1))
                ix2 = max(ix1 + 1, min(w_img, ix2))
                iy2 = max(iy1 + 1, min(h_img, iy2))

                patch = image_arr[iy1:iy2, ix1:ix2]
                if patch.size == 0:
                    samples.append(CellSample(grid.name, r, c, (ix1, iy1, ix2, iy2), empty=True))
                    continue

                hsv = _rgb_to_hsv_array(patch).reshape(-1, 3)
                sat_mask = hsv[:, 1] > config.sat_threshold
                coverage = float(np.mean(sat_mask)) if hsv.shape[0] else 0.0

                if coverage < config.empty_coverage_pct:
                    samples.append(CellSample(
                        grid.name, r, c, (ix1, iy1, ix2, iy2),
                        empty=True, coverage=coverage,
                    ))
                    continue

                fg = hsv[sat_mask]
                hue = _circular_mean_hue(fg[:, 0])
                sat = float(np.median(fg[:, 1]))
                val = float(np.median(fg[:, 2]))
                samples.append(CellSample(
                    grid.name, r, c, (ix1, iy1, ix2, iy2),
                    empty=False, hue=hue, sat=sat, val=val, coverage=coverage,
                ))
    return samples


# ---------------------------------------------------------------------------
# Color clustering
# ---------------------------------------------------------------------------

def _hue_to_unit_circle(hue: np.ndarray) -> np.ndarray:
    """Map hue [0,1] to (cos, sin) on the unit circle so euclidean distance
    on these points = chord length, which approximates angular hue distance."""
    angles = hue * 2.0 * np.pi
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def _chord_for_angle(angle_deg: float) -> float:
    """Chord length on the unit circle for a given angular separation."""
    return 2.0 * float(np.sin(np.radians(angle_deg) / 2.0))


def _cluster_by_hue_gap(hues: np.ndarray, gap_deg: float) -> np.ndarray:
    """Group cells whose hues are within `gap_deg` of each other using DBSCAN
    on the hue circle. Robust against gradient/shadow noise because saturation
    and value are deliberately ignored."""
    if hues.size == 0:
        return np.array([], dtype=np.int32)
    points = _hue_to_unit_circle(hues)
    eps = _chord_for_angle(gap_deg)
    if _HAS_SKLEARN:
        db = DBSCAN(eps=eps, min_samples=1)
        return db.fit_predict(points).astype(np.int32)
    # Fallback: greedy single-link grouping by hue gap.
    order = np.argsort(hues)
    labels = -np.ones(hues.size, dtype=np.int32)
    next_label = 0
    for i in order:
        for j in range(hues.size):
            if labels[j] == -1:
                continue
            if np.linalg.norm(points[i] - points[j]) <= eps:
                labels[i] = labels[j]
                break
        if labels[i] == -1:
            labels[i] = next_label
            next_label += 1
    return labels


# ---------------------------------------------------------------------------
# Pattern: per-grid label arrays + palette
# ---------------------------------------------------------------------------

@dataclass
class Pattern:
    # grid name -> 2D list of labels (strings; EMPTY_LABEL for empty cells)
    grids: dict[str, list[list[str]]] = field(default_factory=dict)
    # label -> {"hue": ..., "sat": ..., "rgb": (r,g,b)}
    palette: dict[str, dict] = field(default_factory=dict)
    # Cell-by-cell sampling diagnostics (for overlay/debug).
    samples: list[CellSample] = field(default_factory=list)
    # Number of distinct non-empty colors.
    k: int = 0

    def to_serializable(self) -> dict:
        return {
            "grids": self.grids,
            "palette": self.palette,
            "k": self.k,
        }


def _hue_sat_to_rgb(hue: float, sat: float, val: float = 0.95) -> tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, max(0.0, min(1.0, sat)), val)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def analyze(image_path: str, config: GridConfig) -> Pattern:
    """Open an image and produce its color-agnostic Pattern."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    samples = sample_cells(arr, config)

    pattern = Pattern()
    for grid in config.grids:
        pattern.grids[grid.name] = [[EMPTY_LABEL] * grid.cols for _ in range(grid.rows)]

    non_empty = [s for s in samples if not s.empty]
    if not non_empty:
        pattern.samples = samples
        return pattern

    hues = np.array([s.hue for s in non_empty])
    labels = _cluster_by_hue_gap(hues, config.hue_gap_deg)

    # Cap at config.max_k by merging the smallest clusters into their nearest
    # neighbor on the hue circle — guards against over-segmentation in noisy
    # images and keeps the permutation search in compare() tractable.
    unique = list(np.unique(labels))
    while len(unique) > config.max_k:
        cluster_hues = {c: _circular_mean_hue(hues[labels == c]) for c in unique}
        cluster_sizes = {c: int(np.sum(labels == c)) for c in unique}
        smallest = min(unique, key=lambda c: (cluster_sizes[c], c))
        candidates = [c for c in unique if c != smallest]
        sx, sy = np.cos(cluster_hues[smallest] * 2 * np.pi), np.sin(cluster_hues[smallest] * 2 * np.pi)
        nearest = min(candidates, key=lambda c: (
            (np.cos(cluster_hues[c] * 2 * np.pi) - sx) ** 2
            + (np.sin(cluster_hues[c] * 2 * np.pi) - sy) ** 2
        ))
        labels = np.where(labels == smallest, nearest, labels)
        unique = list(np.unique(labels))

    k = len(unique)
    pattern.k = k

    # Sort clusters by mean hue for stable label order.
    raw_clusters = list(unique)
    cluster_mean_hue = {c: _circular_mean_hue(hues[labels == c]) for c in raw_clusters}
    order = sorted(raw_clusters, key=lambda c: cluster_mean_hue[c])
    remap = {old: LABEL_ALPHABET[new] for new, old in enumerate(order)}

    for i, s in enumerate(non_empty):
        label = remap[int(labels[i])]
        pattern.grids[s.grid][s.row][s.col] = label

    for new_idx, old_cid in enumerate(order):
        members = [non_empty[i] for i in range(len(non_empty)) if labels[i] == old_cid]
        hue = _circular_mean_hue(np.array([m.hue for m in members]))
        sat = float(np.median([m.sat for m in members]))
        val = float(np.median([m.val for m in members]))
        pattern.palette[LABEL_ALPHABET[new_idx]] = {
            "hue": hue,
            "sat": sat,
            "val": val,
            "rgb": list(_hue_sat_to_rgb(hue, sat, val)),
            "count": len(members),
        }

    pattern.samples = samples
    return pattern


# ---------------------------------------------------------------------------
# Pattern comparison
# ---------------------------------------------------------------------------

@dataclass
class PatternDiff:
    passed: bool
    matched_cells: int
    total_cells: int
    score: float
    # old_label -> new_label
    mapping: dict[str, str] = field(default_factory=dict)
    # List of (grid, row, col, old_label_after_mapping, new_label)
    mismatches: list[tuple[str, int, int, str, str]] = field(default_factory=list)
    error: Optional[str] = None


def _flatten(pattern: Pattern) -> list[tuple[str, int, int, str]]:
    out = []
    for name, grid in pattern.grids.items():
        for r, row in enumerate(grid):
            for c, label in enumerate(row):
                out.append((name, r, c, label))
    return out


def compare(old: Pattern, new: Pattern) -> PatternDiff:
    """Find the best label bijection and report mismatches."""
    old_cells = _flatten(old)
    new_cells = _flatten(new)

    if len(old_cells) != len(new_cells):
        return PatternDiff(False, 0, len(old_cells), 0.0,
                           error=f"Grid size mismatch: {len(old_cells)} vs {len(new_cells)}")

    old_labels = sorted({lbl for *_, lbl in old_cells if lbl != EMPTY_LABEL})
    new_labels = sorted({lbl for *_, lbl in new_cells if lbl != EMPTY_LABEL})

    # Pair cells positionally.
    paired = list(zip(old_cells, new_cells))

    # Empty cells map to themselves; only permute the non-empty old labels.
    if len(old_labels) == 0 or len(new_labels) == 0:
        # Degenerate case: just compare directly.
        mismatches = [(g, r, c, ol, nl) for (g, r, c, ol), (_, _, _, nl) in paired if ol != nl]
        matched = len(paired) - len(mismatches)
        return PatternDiff(
            passed=(len(mismatches) == 0),
            matched_cells=matched,
            total_cells=len(paired),
            score=matched / max(1, len(paired)),
            mapping={},
            mismatches=mismatches,
        )

    # Brute-force best mapping. Pick max(k_old, k_new) permutations over the larger set.
    if len(old_labels) <= len(new_labels):
        from_labels, to_labels = old_labels, new_labels
        flip = False
    else:
        from_labels, to_labels = new_labels, old_labels
        flip = True

    best_score = -1
    best_mapping: dict[str, str] = {}
    best_mismatches: list[tuple[str, int, int, str, str]] = []

    # Cap labels (k! grows fast); 8! = 40320 is fine.
    if len(to_labels) > 8:
        to_labels = to_labels[:8]

    for perm in permutations(to_labels, len(from_labels)):
        mapping = dict(zip(from_labels, perm))
        matched = 0
        mismatches: list[tuple[str, int, int, str, str]] = []
        for (g, r, c, ol), (_, _, _, nl) in paired:
            if flip:
                mapped_new = mapping.get(nl, nl)
                if ol == mapped_new or (ol == EMPTY_LABEL and nl == EMPTY_LABEL):
                    matched += 1
                else:
                    mismatches.append((g, r, c, ol, mapped_new))
            else:
                mapped_old = mapping.get(ol, ol)
                if mapped_old == nl or (ol == EMPTY_LABEL and nl == EMPTY_LABEL):
                    matched += 1
                else:
                    mismatches.append((g, r, c, mapped_old, nl))
        if matched > best_score:
            best_score = matched
            best_mapping = mapping
            best_mismatches = mismatches
            if matched == len(paired):
                break

    if flip:
        # Express mapping consistently as old -> new.
        inv = {v: k for k, v in best_mapping.items()}
        best_mapping = inv

    total = len(paired)
    return PatternDiff(
        passed=(best_score == total),
        matched_cells=best_score,
        total_cells=total,
        score=best_score / max(1, total),
        mapping=best_mapping,
        mismatches=best_mismatches,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _label_color(label: str, pattern: Pattern) -> tuple[int, int, int]:
    if label == EMPTY_LABEL:
        return (60, 60, 70)
    if label in pattern.palette:
        return tuple(pattern.palette[label]["rgb"])
    idx = LABEL_ALPHABET.find(label)
    return DISPLAY_PALETTE_RGB[idx % len(DISPLAY_PALETTE_RGB)] if idx >= 0 else (200, 200, 200)


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def render_label_grid(pattern: Pattern, cell_px: int = 48, title: str = "") -> Image.Image:
    """Render the label grids as a labeled image."""
    grids = list(pattern.grids.items())
    if not grids:
        return Image.new("RGB", (200, 80), (30, 30, 50))

    pad = 16
    title_h = 28 if title else 0
    grid_pad = 18
    widths = [pad + cols * cell_px + pad
              for _, g in grids for cols in [len(g[0]) if g else 0]]
    heights = [grid_pad + len(g) * cell_px + grid_pad for _, g in grids]

    W = max(widths)
    H = title_h + sum(heights) + (len(grids) - 1) * 8 + pad
    img = Image.new("RGB", (W, H), (20, 22, 35))
    draw = ImageDraw.Draw(img)

    font = _font(int(cell_px * 0.42))
    title_font = _font(18)

    if title:
        draw.text((pad, 6), title, fill=(220, 220, 230), font=title_font)

    y_off = title_h
    for name, grid in grids:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        gw = cols * cell_px
        gh = rows * cell_px
        gx = (W - gw) // 2
        gy = y_off + grid_pad

        draw.text((gx, gy - 16), f"{name} ({rows}x{cols})", fill=(140, 140, 160), font=_font(12))

        for r in range(rows):
            for c in range(cols):
                label = grid[r][c]
                color = _label_color(label, pattern)
                x1, y1 = gx + c * cell_px, gy + r * cell_px
                x2, y2 = x1 + cell_px - 2, y1 + cell_px - 2
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(15, 15, 25), width=2)
                if label != EMPTY_LABEL:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    cx = x1 + (cell_px - 2 - tw) // 2 - bbox[0]
                    cy = y1 + (cell_px - 2 - th) // 2 - bbox[1]
                    # Outlined for legibility on any color.
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        draw.text((cx + dx, cy + dy), label, fill=(0, 0, 0), font=font)
                    draw.text((cx, cy), label, fill=(255, 255, 255), font=font)
        y_off = gy + gh
    return img


def render_side_by_side_labels(old: Pattern, new: Pattern, diff: PatternDiff) -> Image.Image:
    """Old labels | New labels, side by side, with the mapping below."""
    cell_px = 44
    left = render_label_grid(old, cell_px=cell_px, title="OLD pattern")
    right = render_label_grid(new, cell_px=cell_px, title="NEW pattern")

    pad = 18
    W = left.width + right.width + pad * 3
    H = max(left.height, right.height) + 90
    canvas = Image.new("RGB", (W, H), (15, 15, 25))
    canvas.paste(left, (pad, 0))
    canvas.paste(right, (pad * 2 + left.width, 0))

    draw = ImageDraw.Draw(canvas)
    status_color = (76, 175, 80) if diff.passed else (244, 67, 54)
    label = "PATTERN MATCH" if diff.passed else f"PATTERN MISMATCH ({diff.total_cells - diff.matched_cells} cell(s) differ)"
    draw.text((pad, max(left.height, right.height) + 8), label, fill=status_color, font=_font(16))

    mapping_str = ", ".join(f"{k}→{v}" for k, v in sorted(diff.mapping.items()))
    if mapping_str:
        draw.text((pad, max(left.height, right.height) + 36), f"color mapping: {mapping_str}",
                  fill=(170, 170, 200), font=_font(13))
    return canvas


def render_mismatch_overlay(image_path: str, pattern: Pattern, diff: PatternDiff,
                            config: GridConfig) -> Image.Image:
    """Dim the screenshot and outline mismatched cells in red."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    # Dim everything outside the mismatched cells.
    dim = (arr.astype(np.float32) * 0.35).astype(np.uint8)
    out = Image.fromarray(dim)
    draw = ImageDraw.Draw(out, "RGBA")

    sx, sy = _scale_factors(arr, config)
    mismatch_set = {(g, r, c) for g, r, c, *_ in diff.mismatches}

    # Outline each grid's cells faintly; mismatched cells get a bright red.
    for grid in config.grids:
        gx = grid.x * sx
        gy = grid.y * sy
        cw = (grid.width * sx) / grid.cols
        ch = (grid.height * sy) / grid.rows
        for r in range(grid.rows):
            for c in range(grid.cols):
                x1 = int(gx + c * cw)
                y1 = int(gy + r * ch)
                x2 = int(gx + (c + 1) * cw)
                y2 = int(gy + (r + 1) * ch)
                if (grid.name, r, c) in mismatch_set:
                    # Reveal the original pixels inside the mismatched cell.
                    out.paste(img.crop((x1, y1, x2, y2)), (x1, y1))
                    draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 50, 50, 255), width=3)
                else:
                    draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(80, 80, 100, 90), width=1)
    return out


def render_pattern_overlay(image_path: str, pattern: Pattern, config: GridConfig) -> Image.Image:
    """Show the assigned label on top of each cell — useful for debugging clustering."""
    img = Image.open(image_path).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    arr = np.array(img)
    sx, sy = _scale_factors(arr, config)
    font = _font(max(12, int(min(arr.shape[:2]) / 48)))

    for grid in config.grids:
        gx = grid.x * sx
        gy = grid.y * sy
        cw = (grid.width * sx) / grid.cols
        ch = (grid.height * sy) / grid.rows
        rows = pattern.grids.get(grid.name, [])
        for r in range(grid.rows):
            for c in range(grid.cols):
                label = rows[r][c] if r < len(rows) and c < len(rows[r]) else EMPTY_LABEL
                x1 = int(gx + c * cw)
                y1 = int(gy + r * ch)
                x2 = int(gx + (c + 1) * cw)
                y2 = int(gy + (r + 1) * ch)
                draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 255, 255, 80), width=1)
                if label != EMPTY_LABEL:
                    color = _label_color(label, pattern)
                    draw.rectangle([x1, y1, x1 + 22, y1 + 18], fill=color + (220,))
                    draw.text((x1 + 5, y1 + 1), label, fill=(0, 0, 0), font=font)
    return img
