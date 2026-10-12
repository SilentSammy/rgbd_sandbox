# Region OCR demo

Run `python ocr_benchmark_demo.py` to compare RapidOCR and EasyOCR on the five
tabletop object crops in `recordings/20261005_195417.db3`, frame 137.
Results and original-resolution RGB crops are saved in `ocr_benchmark/`.
Each crop is enlarged 2x and tested at four orientations. Model loading is
timed separately from inference; inference timings include detection and recognition.

RapidOCR now uses the faster path in `rgbd_ocr.py` by default. It avoids the
engine's automatic upscale to a 736-pixel **shorter** side, detects text at two
perpendicular crop orientations, rectifies each detected text line, and tests
both reading directions. Recognition is batched across all objects. Neither
package orientation nor the direction classifier is assumed to be correct.
The interactive demos and MP4 exporter now request all four directions.

`RegionOCR("rapidocr", fast=False)` retains the original exhaustive pipeline
for comparison. EasyOCR is unchanged. Fast results retain per-region direction
hypotheses and full-image text polygons; opposite directions share detection
polygons. `rotation` labels a detection-axis/reading-direction hypothesis, not
a measured text angle. Shared detector/recognizer time is apportioned across
results; use wall-clock timing around `read_regions()` for complete latency.

For a reproducible speed comparison and eight-angle orientation checks:

```powershell
python ocr_speed_benchmark.py --frames 137 131 180 --repeats 3
```

This saves `ocr_benchmark/speed_results.json`, comparing original, resize-only,
and batched pipelines. It times OCR after warmup, excluding model loading and
segmentation. Dictionary matches are reported separately from raw OCR so missed
labels and conflicting suggestions can be inspected. It does not establish
medicine identity, dosage, or general accuracy on unseen packaging.

Measured on five segmented objects in `20261005_195417.db3`, with all four
directions enabled and three timed repeats: frame 137 took a median 6,670 ms
originally, 781 ms with resizing alone corrected, and 525 ms with batching.
Frame 180 went from 6,505 ms to 651 ms. The same dictionary labels—Melatonina,
Ibuprofeno, and Colgate—were retained on the three checked frames and at all
eight tested angles of frame 137. The other two objects remain unidentified.
Raw readings can differ even when these dictionary matches agree.

This is roughly 10–13x faster in those two frame benchmarks, not 30 FPS OCR.
Keeping 2x crop enlargement was intentional: a separate native-resolution
probe missed Colgate at one quarter-turn. CPU timing varies with load;
`speed_frame131_confirmed.json` contains a separate frame-131 confirmation run.
Use `--angles` alone to skip rotated-crop tests for a quicker timing run.

For interactive navigation with segmentation and OCR:

```powershell
python segmentation_ocr_demo.py --start-frame 137
```

Focus an image window: `A` reverse, `S` pause, `D` play, `Q` previous frame,
`E` next frame, `W` restart at `--start-frame`, and Escape to quit. All interactive
demos share these controls from `rgbd_db3_viewer.py`. Playback starts paused;
stepping/restarting pauses it, and it stops at either end without looping.
Playback targets the recording's FPS without skipping frames, so heavy processing
may slow it down. OCR runs in a background
worker so navigation stays responsive; stale results cannot replace the current
frame. The last eight completed frames are cached. Use `--max-dimension 800`
to resize the complete image/panel window.

For known-label matching:

```powershell
python label_matching_demo.py --start-frame 137
```

Edit `LABELS` in `ocr_label_setup.py` to change the vocabulary and legitimate
aliases. The demo preserves raw OCR and adds dictionary suggestions underneath.
It normalizes accents/case/punctuation and uses normalized Levenshtein distance.
Defaults require a similarity of at least 0.75 and a lead of at least 0.15 over
the next canonical label. Short fragments are ignored; weaker candidates are
marked uncertain or unknown. Matching lowers the OCR display cutoff to 0.3 so
uncertain raw readings can be inspected; the original viewer retains its 0.8 cutoff.
OCR confidence and dictionary similarity are distinct scores, not probabilities.
No product identity, strength, or dosage is inferred from a label suggestion.

Run `python -m unittest test_rgbd_label_matching test_rgbd_ocr` for matcher and
OCR-coordinate tests.

For a manually selected RGB region (right/bottom coordinates are exclusive):

```powershell
python ocr_benchmark_demo.py --engines rapidocr --roi 100 200 300 500
```

The reusable API accepts the reader's RGB arrays:

```python
from rgbd_ocr import RegionOCR

ocr = RegionOCR("rapidocr")  # Initialize once, then reuse.
results = ocr.read_regions(frame.color, [(100, 200, 300, 500)])
```

Results include text, engine confidence, text polygons in original full-image
coordinates, tested crop orientation (counterclockwise), and elapsed time.
Rotations are returned separately, rather than silently combining conflicting readings.
Confidence is not calibrated across engines or a guarantee of correct spelling.
Added pixels inside the object rectangles are not proof that text belongs to that object.

## Installation

RapidOCR is included in `requirements.txt`; its ONNX models ship with the package.
EasyOCR is an optional, heavier comparison engine:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install scipy scikit-image python-bidi ninja pyclipper shapely PyYAML Pillow
python -m pip install easyocr==1.7.2 --no-deps
```

The final `--no-deps` deliberately keeps the existing **GUI-enabled**
`opencv-python`, rather than installing EasyOCR's `opencv-python-headless`
dependency over the viewers' OpenCV. Pip may report that missing metadata
dependency; the existing `cv2` provides the required functionality.
EasyOCR downloads weights into `.ocr_models/easyocr/` on its first run.
Subsequent runs are local/offline. No images are uploaded.

Run `python -m unittest test_rgbd_ocr` for crop/rotation coordinate tests.
