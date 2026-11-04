# Medicine segmentation comparison

Separate experiment comparing SAM 2.1 RGB masks with our actual tabletop depth
labels. Inspired by Hello Robot's Stretch 4 grasping demo; see ATTRIBUTION.md.
The existing live robot project is not modified or restarted by this experiment.

See [RESULTS.md](RESULTS.md) for verified sample galleries, measured timings,
and known mask failures from the first runs.

This first version focuses on delineating medicine packages. It does not identify
drug names/doses, validate medicine identity, or command the robot. Molmo is
deliberately omitted: use clicks to isolate SAM quality, or depth proposals for
an automatic paired comparison. Depth proposals inherit the baseline's missed
objects/false positives; use clicks to test an object on the second shelf when
the depth baseline misses it.

## Run on this laptop

From the enclosing `rgbd_sandbox` directory:

```powershell
medicine_segmentation_compare\.venv\Scripts\python.exe medicine_segmentation_compare\demo.py --start-frame 137
```

Two 600-pixel windows show actual masks. Click in the SAM window to add a
foreground point; clicks replace automatic depth boxes. C/B clears clicks and
restores depth-box prompts. Shared playback: A reverse, S pause, D play,
Q/E step, W restart, Escape exit. G saves an additional snapshot.
CPU inference can take seconds; interactive events are handled between inference
calls. GUI playback is an inspection tool, not a real-time claim. Native RGB and
depth calibration are preserved; rotation defaults to one clockwise turn.

```powershell
# One reproducible benchmark frame, no GUI; all artifacts saved automatically.
medicine_segmentation_compare\.venv\Scripts\python.exe medicine_segmentation_compare\demo.py --headless --frames 1 --start-frame 137

# Same model size as Hello Robot; requires substantially more compute.
medicine_segmentation_compare\.venv\Scripts\python.exe medicine_segmentation_compare\demo.py --model facebook/sam2.1-hiera-large

# A selected object, independent of the tabletop proposals (native RGB pixels).
medicine_segmentation_compare\.venv\Scripts\python.exe medicine_segmentation_compare\demo.py --points "170,550" --headless

# Short sequence: re-prompt SAM with its previous boxes on consecutive frames.
medicine_segmentation_compare\.venv\Scripts\python.exe medicine_segmentation_compare\demo.py --headless --frames 10 --track-boxes

# Inspect a debug bundle retrieved from the robot, including its frame history.
medicine_segmentation_compare\.venv\Scripts\python.exe medicine_segmentation_compare\demo.py path\to\debug_snapshot --start-frame 0
```

`--live --serial SERIAL` optionally reads a locally attached RealSense using the
existing SDK adapter (install pyrealsense2 in this project's venv first). It does
not connect to the robot over the network. Use recorded snapshots for comparison
without taking ownership of the robot camera. `--baseline-only` is an explicitly
labeled diagnostic mode that never substitutes a fake SAM result.

Default: Tiny, automatically CUDA if available, otherwise CPU. `--device cuda`
requires a CUDA-enabled PyTorch installation. First run downloads weights into
`models/`; later runs reuse them. The project's venv is isolated from the existing
robot/bag demo environment. Create it on another machine with:

```powershell
python -m venv medicine_segmentation_compare\.venv
medicine_segmentation_compare\.venv\Scripts\python.exe -m pip install -r medicine_segmentation_compare\requirements.txt
```

The default Tiny revision is pinned to the checkpoint verified in RESULTS.md;
`--revision` overrides it. The local CPU PyTorch/torchvision files were reused
from the existing installed distributions by copying them into this isolated
venv; the other dependencies were installed with pip. No system packages were
changed. Use a CUDA-enabled PyTorch installation for GPU testing; `--device cuda`
cannot enable CUDA in a CPU-only PyTorch build.

The project imports our source/calibration/segmentation/viewer modules from its
parent folder. `--max-object-depth` defaults to the baseline's present 1.05 m
centroid cutoff. `--segmentation-scale .5` matches the current default. Both are
recorded in exports, and the parent modules are hashed for provenance.

## Inspect outputs

Every computation creates a uniquely named frame/visit folder under
`outputs/<run>/`; repeated visits never overwrite previous results. Open
`index.html` for an image gallery with measurement links. Each frame contains:

- Raw RGB and floating-point metric depth, full calibration and timestamps.
- Native depth labels, projected RGB baseline labels, and separate boolean SAM
  masks (overlapping predictions are preserved rather than overwritten).
- Mask PNGs, per-object RGB crops for future OCR, individual overlays, and a
  side-by-side `comparison.png`.
- `report.json`: prompts and their source, model/version/device, predicted quality,
  pixel area, boxes, median/range of measured depths, measured 3D center/extents,
  pairwise mask agreement, support-plane failure/status, and processing timings.

Use `--save-interval 5` during live inspection to limit automatic bundles to one
every five seconds; clicks and G can still save additional results. The default
interval is zero so offline benchmarks retain every processed frame. Playback
keys apply to bags/snapshots; live mode uses Escape to exit.

`summary.jsonl` summarizes frame and export timings; model inference times exclude
model loading, acquisition, display, and disk exports. `config.json` records baseline parameters
and source hashes; `environment.json` records dependencies/model revision. Errors
produce `traceback.txt` and, when a frame was acquired, a replayable `failure_input/`
bundle. Runs are intentionally not auto-restarted so failures
remain visible during benchmarking. Each exported full-resolution frame can use
several MB; use a bounded `--frames` batch for unattended comparisons.

Agreement IoU compares SAM to the depth mask only where native depth projects to
RGB (nearest color-camera Z wins). It is not accuracy: SAM can correctly fill
depth holes, or incorrectly include background. Model quality is a prediction,
not measured ground-truth IoU. 3D measurements use only available, z-buffered
depth samples within a SAM mask; they do not reconstruct the object's unseen back.

`--track-boxes` is the simple image-mode alternative used conceptually in the
upstream code: previous mask bounds become the next frame's prompts. It can drift,
and resets on jumps/reversals. This experiment has no board-based persistent IDs
or SAM video memory. Each frame's fresh depth-proposal order may change; displayed
mask indices must not be interpreted as object identities. Manual clicks without
`--track-boxes` stay at the same image coordinates across frames.

To establish which method is better, manually label identical RGB frames and
compare object misses/merges, background leakage, and latency. No ground-truth
accuracy claim is made from mask agreement alone.
