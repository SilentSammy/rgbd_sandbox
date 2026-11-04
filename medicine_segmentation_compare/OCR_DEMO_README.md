# Persistent object OCR demo

`live_ocr_demo.py` uses the same board-anchored table plane, coarse depth boxes
and persistent IDs as the live SAM experiment. RapidOCR runs in one background
worker, including model initialization. It batches the optimized text detection
and recognition path across eligible objects and tests both perpendicular axes
and reading directions. No package orientation is assumed.

Appearance-assisted ID recovery is enabled by default; see
`../VISUAL_TRACKING_README.md`. This uses masked RGB signatures, not SAM or
continuous optical flow. Missing confirmed IDs retain their labels for 30
capture seconds. `--no-appearance --forget-after 5` restores the old baseline.
`[R1]` beside an ID means it has been recovered once.

The demo limits measured tabletop footprint diameter to 20 cm by default
(`--max-footprint .20`; `0` disables). Height is excluded and the measurement is
invariant to camera rotation. This removes the 23.1 cm wall patch in the ID 229
snapshot while retaining the six packages (4.7–9.2 cm measured footprints).
The low-level segmenter's optional `max_object_footprint` defaults to None,
preserving depth-only callers' prior behavior. G reports now include the exact
camera-frame plane, candidate extents/footprints and track detection indices.

One job is outstanding at a time. Each job retains its original RGB, inclusive
object box (converted to exclusive padded OCR crop coordinates), persistent ID,
reference anchor, source timestamp, and accumulated capture-time clock. Delayed
text attaches to that ID if it still exists and its reference position agrees
within 8 cm. Old text polygons are saved with the source image; they are never
drawn on a different frame. Current boxes/IDs continue updating independently.
Clearing memory invalidates pending results. Refresh defaults to five seconds
of processed video time, not wall time. R forces a new reading on a paused frame.

The object window shows current boxes and IDs. A separate 600-pixel panel shows
raw OCR/confidence and dictionary suggestions per ID, with result age measured
in accumulated capture time. The strongest unambiguous accepted label is kept
for each ID: higher dictionary similarity wins, then OCR confidence, then match
margin. Equal quality keeps the earlier evidence. Empty, ambiguous or weaker
readings retain that label; strictly stronger evidence can replace it.
The panel shows the remembered label, its original confidence/age, and the latest
raw OCR/match separately. Conflicting evidence is flagged. Without an accepted
historical match, ambiguous readings display AMBIGUOUS and other misses Unknown.
Labels survive temporary missing observations while the tracker retains the ID;
forgetting that ID or pressing C discards them. This is session memory, not ID
restoration across program restarts. OCR confidence and dictionary similarity
are different scores. Labels are text-based suggestions, not verified medicine
identities, strengths or dosages. Edit `../ocr_label_setup.py` for vocabulary.

## On the robot, once cameras are available

```bash
cd /home/hello-robot/stretch_workspace/rgbd_sandbox/medicine_segmentation_compare
bash run_live_ocr.sh
```

The launcher uses the current close-range scene settings. Optional SAM can run
alongside OCR to refine crop boxes:

```bash
bash run_live_ocr.sh --sam
```

SAM and OCR are asynchronous; SAM uses a separate process and OCR uses a native
ONNX-backed worker thread (two inference threads by default). OCR initially uses
coarse boxes, and requests a fresh reading when SAM refinements arrive. OCR
then uses an enclosing box covering both the coarse and refined bounds:
measured SAM geometry can omit text around depth holes, so narrowing the crop
to that geometry alone lost Colgate in an offline probe. This
combined CPU workload still needs a live performance check. The OCR-only launch
avoids requiring SAM for label recognition.

## Offline, without opening any camera

By default the script opens the benchmark bag at frame 137, paused:

```powershell
.venv\Scripts\python.exe medicine_segmentation_compare/live_ocr_demo.py
```

Or run from the experiment folder on the robot:

```bash
env -u PYTHONPATH .venv/bin/python live_ocr_demo.py \
  outputs/flicker/live_check --start-frame 0 --scene-depth 1.1 --plane-min-inliers 300
```

Inputs can be a DB3 bag, a replayable debug snapshot, or the saved flicker-capture
directory containing `sequence.npz` and `calibration.json`. Only `--live` imports
the camera adapter and opens a RealSense. Offline inspection accepts IDs
immediately so a single paused frame can be read. Live mode retains the one-second
probation period. `--probation-seconds` overrides either setting.

For reproducible processing, including waiting for the final OCR job:

```bash
env -u PYTHONPATH .venv/bin/python live_ocr_demo.py \
  outputs/flicker/live_check --headless --frames 60 --start-frame 0 \
  --scene-depth 1.1 --plane-min-inliers 300
```

Recorded playback reuses `Playback` from `rgbd_db3_viewer.py`: A reverse,
S pause, D play, Q/E step, W restart, Escape quit. Live S pauses/resumes the view;
Q/Escape exits. R rereads, C clears IDs/OCR, T relearns the table, G saves a
snapshot. Worker completions remain visible while recorded playback is paused.

Outputs: `outputs/ocr/<run>/`. Each OCR job saves source RGB, individual crops
named by persistent ID, all direction hypotheses and original text polygons,
raw readings, dictionary evidence, capture timestamp and inference time.
`latest.json` and `events.jsonl` report current associations, latest readings and
`best_match` evidence with its capture timestamp. Per-job reports remain the
raw readings from that capture. G also saves both
display panels, raw RGB/depth, calibration and pose for replay/debugging. Linux
SIGUSR1 requests the same snapshot. Snapshot disk writes can briefly pause the UI.

RapidOCR 1.4.4 ships its ONNX models with the package. Installation in the
experiment venv must preserve the existing GUI-enabled OpenCV:

```bash
.venv/bin/python -m pip install rapidocr_onnxruntime==1.4.4 --no-deps
```

Its dependencies must already be present (as verified on this robot); otherwise
install missing dependencies into that venv. Existing SAM dependencies remain
in `requirements.txt`; OCR is pinned separately in `requirements-ocr.txt`.

## Verified before live testing

- All 79 root tests passed, including five new delayed-ID/crop/scheduling tests.
- The recorded GUI completed OCR while paused, rendered both windows, and
  exited cleanly without opening a camera. Frame 137 matched Melatonina,
  Ibuprofeno and Colgate; two other objects remained unknown.
- A 60-frame replay of the robot's captured RGB-D sequence completed OCR for
  six persistent IDs. Ibuprofeno, Diclofenaco, Melatonina and Colgate matched;
  BUTTER and Sepramycina stayed raw/unknown because they are outside the vocabulary.
- The camera-free SAM+OCR test completed both coarse and depth+SAM crop readings,
  retaining those four known labels after the crop fix. Robot evidence:
  `outputs/ocr/20261103_141130_012910/`. The separate 60-frame evidence is
  `outputs/ocr/20261103_140628_261092/`.
- Robot OCR batches ranged from about 0.7 to 4 seconds in these cold/warm and
  concurrent-load probes. This is background processing, not real-time OCR.
  Live capture and combined CPU responsiveness remain to be checked when the
  cameras become available. No cameras were opened during these setup tests.
