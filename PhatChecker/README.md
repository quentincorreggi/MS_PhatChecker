# Visual Regression Diff Tool

A tool for comparing game level screenshots side by side to catch visual regressions. It uses SSIM (Structural Similarity Index) to detect differences between an "old" and "new" set of screenshots, highlights what changed, and generates an interactive HTML report.

It comes with two interfaces: a **web UI** (recommended for most users) and a **CLI** for scripting or CI pipelines.

---

## Quick Start

### Prerequisites

- Python 3.8 or later

### Installation & Launch

```bash
# Clone or copy this folder, then:
./run.sh
```

That's it. On first run, the script creates a virtual environment, installs all dependencies, and opens the web UI at **http://localhost:5005**. On subsequent runs it starts instantly.

> If `./run.sh` doesn't work, try `bash run.sh` or make it executable first with `chmod +x run.sh`.

### Manual Installation (if you prefer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 visual_diff_app.py
```

---

## Web UI

Open **http://localhost:5005** after launching. The interface has three tabs:

### Setup

1. **Old Screenshots** — folder containing the "before" PNGs (the reference).
2. **New Screenshots** — folder containing the "after" PNGs (the version to test).
3. **Output Directory** — where the report and diff images will be saved. Leave empty to auto-generate next to the old folder.
4. **SSIM Threshold** — how strict the pass/fail check is. Default `0.995` (very strict). Lower it if you expect minor acceptable changes.
5. **Workers** — number of parallel processes. Increase for large batches, decrease if your machine struggles.
6. **Save pass diffs** — check this to also generate diff images for levels that passed (uses more disk space).

Click **Run Comparison** and wait for it to finish. The tool will match files across both folders by level name (e.g. `Level30_Easy.png` matches `Level30_Hard.png` since both share the `Level30` key).

### Mask Editor

Use this to exclude or compensate for regions that are expected to differ (animated elements, UI overlays, shifted content).

**Exclusion Masks** (blue) — draw rectangles over areas that should be completely ignored during comparison (e.g. an animated score counter, a timer, particle effects).

**Shift Regions** (orange) — draw rectangles over areas where content has moved by a known number of pixels in the new screenshots. After drawing, enter the X/Y pixel offset. The tool realigns those regions before comparing, so a simple position change won't be flagged as a regression.

You can:
- Load a sample image from the Old folder or upload one manually.
- Switch between the Mask and Shift tools using the buttons above the canvas.
- Undo, clear, or remove individual regions.
- Save the configuration to a JSON file (reusable from CLI too).

### Results

After a comparison finishes, this tab shows a summary (total, passed, failed, missing) and a list of all levels sorted by score. Click any failed level to expand it and see:

- **Old / New thumbnails** — quick side-by-side preview.
- **Side by Side** — old, new, and diff next to each other.
- **Highlighted Changes** — changed pixels in red with a yellow outline, unchanged pixels dimmed.
- **Amplified Diff** — raw pixel differences amplified 5x so subtle changes pop.

You can also open the full HTML report directly from the output folder.

---

## CLI Usage

For scripting, CI pipelines, or if you prefer the terminal.

### Compare two folders

```bash
python3 visual_diff.py old_screenshots/ new_screenshots/
```

### Compare with a mask config

```bash
python3 visual_diff.py old/ new/ --mask mask_config.json
```

### Compare a single pair of images

```bash
python3 visual_diff.py --single old/level_190.png new/level_190.png
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `old` | *(required)* | Path to old screenshots folder (or old image with `--single`) |
| `new` | *(required)* | Path to new screenshots folder (or new image with `--single`) |
| `--output-dir`, `-o` | `regression_report/` | Where to save the report and diff images |
| `--threshold`, `-t` | `0.995` | SSIM threshold for pass/fail (0 to 1, higher = stricter) |
| `--single` | off | Compare a single pair instead of batch |
| `--workers`, `-w` | `4` | Number of parallel worker processes |
| `--save-pass` | off | Generate diff images for passed levels too |
| `--mask`, `-m` | none | Path to a mask config JSON file |

---

## File Matching Logic

The tool matches old and new screenshots by extracting a **level key** from each filename:

- `Level30_VeryHard_30_9C_TH.png` → key: `level30`
- `Level30_Easy_Level.png` → key: `level30`
- `level_190.png` → key: `level_190`

This means filenames don't need to be identical — as long as they share the same level number prefix, they'll be paired. Files that exist in only one folder are reported as "missing".

---

## Mask Config Format

The mask config is a JSON file (created via the web UI's Mask Editor or written by hand):

```json
{
  "imageWidth": 1080,
  "imageHeight": 1920,
  "masks": [
    { "x": 50, "y": 100, "width": 200, "height": 80 }
  ],
  "shifts": [
    { "x": 0, "y": 500, "width": 1080, "height": 400, "shiftX": 0, "shiftY": -20 }
  ]
}
```

- `imageWidth` / `imageHeight` — the resolution at which masks were drawn. Coordinates are automatically scaled if actual screenshots differ in size.
- `masks` — regions to ignore entirely.
- `shifts` — regions where content has shifted by a known offset. `shiftX` / `shiftY` are how many pixels the region moved **right** / **down** in the new screenshots.

---

## Output

After a batch run, the output directory contains:

```
regression_report/
├── report.html          ← interactive HTML report (open in browser)
└── diffs/
    ├── level30_highlighted.png
    ├── level30_amplified.png
    ├── level30_sidebyside.png
    ├── level30_ssim.png
    ├── level30_thumb_old.png
    ├── level30_thumb_new.png
    └── ...
```

---

## How SSIM Works (in brief)

SSIM compares two images based on luminance, contrast, and structure. A score of `1.0` means the images are identical. The default threshold of `0.995` means anything below 99.5% similarity is flagged as a failure. This is intentionally strict — for game levels, even small unintended changes matter.

---

## Troubleshooting

**"No matching PNG files found"** — make sure both folders contain `.png` files with matching level names. The tool only looks at PNG files.

**Images have different sizes** — the tool automatically resizes the new image to match the old one and adds a warning in the report. No action needed, but it's worth checking if this is expected.

**Too many false positives** — try lowering the threshold (e.g. `0.98`) or use the Mask Editor to exclude dynamic regions like animated UI, particle effects, or timers.

**Port 5005 already in use** — another instance is probably still running. Kill it or wait for it to stop, then try again.
