# RGB-D detection and classification

Source-independent detection with separate pose providers and background OCR.
The previous code remains in `../rgbd_sandbox_legacy`.

## Setup and demos

```powershell
python -m pip install -r requirements.txt
# Optional classification dependencies:
python -m pip install -r requirements-ocr.txt

.venv\Scripts\python.exe detection_pipeline.py --start-frame 137
.venv\Scripts\python.exe classification.py --start-frame 137
```

Every demo defaults to
`../rgbd_sandbox_legacy/recordings/20261005_195417.db3`. An optional first
positional argument selects another bag. `--rotations 1` rotates images and
calibration clockwise together. `--max-dimension` defaults to 600.

Playback is centralized in `rgbd_demo.py`: A reverse, S pause, D play, Q/E
previous/next, W restart at `--start-frame`, Escape quit. Demos start paused.
There is no settings GUI. Background OCR refreshes a paused display without
reprocessing detection or advancing lifecycle timers.

## Components

| Module / class | Responsibility |
| --- | --- |
| `rgbd_db3_reader.RealSenseReader` | RealSense Viewer DB3 -> synchronized frames and calibration, without ROS |
| `board_pose_estimator.BoardPoseEstimator` | RGB frame + known board -> raw shared pose, marker diagnostics |
| `pose_filter.PoseFilter` | Pose -> short-window position/rotation mean |
| `tabletop_segmenter.TabletopSegmenter` | Depth + optional pose -> masks, boxes and geometry; remembers only the support plane |
| `appearance.RGBObjectAppearance` | Frame + detections -> masked ORB/color signatures and RGB crop boxes; no galleries |
| `object_persistor.ObjectPersistor` | Geometry + pose + appearance + capture dt -> persistent ID snapshots and association events |
| `detection_pipeline.DetectionPipeline` | Compose segmentation, appearance and persistence; expose status and stage timings |
| `ocr_reader.OCRReader` | Synchronous RGB crops -> text, confidence, direction hypotheses and source-image polygons |
| `product_matcher.ProductMatcher` | Text -> dictionary evidence or ambiguity; no temporal memory |
| `classification_memory.ClassificationMemory` | Retain strongest accepted product evidence per persistent ID |
| `classification.ObjectClassifier` | One bounded OCR worker, capture-time scheduling and checked delayed-result association |

Acquisition, pose estimation/filtering, OCR and GUI stay outside detection:

```python
from board_pose_estimator import BoardPoseEstimator
from pose_filter import PoseFilter
from detection_pipeline import DetectionPipeline

estimator = BoardPoseEstimator(calibration.color)
smoother = PoseFilter()
pipeline = DetectionPipeline(calibration)
pose = smoother.update(estimator.estimate(frame))
result = pipeline.process(frame, pose)
```

An odometry/SLAM adapter can supply `pose` directly. Live RealSense and ROS2
adapters need only supply the shared types; those adapters are not ported yet.
`FrameSource` requires calibration and frame iteration; indexing/count/FPS are
playback extensions. The DB3 reader supports the recorded RealSense Viewer
layout, not arbitrary ROS topic names.

## Shared contracts

`rgbd_types.py` has only NumPy and standard-library dependencies:

- `Frame`: RGB uint8 HxWx3, floating-point depth in meters, integer capture time
  in nanoseconds. Zero/NaN depth is invalid. Depth is optical Z, not radial range.
- `Camera` / `Calibration`: native dimensions, intrinsics/distortion, and rigid
  `depth_to_color` in meters. Images and calibration must share orientation.
- `Pose`: timestamped rigid `reference_to_camera` into the **RGB optical camera**,
  with inverse `camera_to_reference`. Optical axes: X right, Y down, Z forward.
  A robot-base pose must first be converted to a camera pose by its adapter.
- `Detection`: native-depth boolean mask, depth-camera centroid/XYZ extents in
  meters, inclusive padded depth box, visible footprint diameter and peak height.
- `Track`: non-recycled instance ID, position in fixed-reference meters, current
  detection index (None when missing), confirmation and lifecycle state.

For homogeneous XYZ1 column vectors:

```python
rgb_point = calibration.depth_to_color @ depth_point
reference_point = pose.camera_to_reference @ rgb_point
```

Call the pipeline serially with matching pose/frame capture timestamps.
Live timestamps must increase. `allow_reverse=True` is for browsing: lifecycle
age uses abs(capture dt), never wall-clock processing time. Reference changes
clear incompatible tracks/planes. `pipeline.reset()` clears history and elapsed
time while preserving the ID counter.

## Mutable settings and geometric behavior

Parameters are public dataclasses, read on each processing call:

```python
pipeline.segmenter.parameters.max_object_footprint = .20
pipeline.segmenter.plane_parameters.max_rms = .004
pipeline.extractor.parameters.features = 120
pipeline.persistor.parameters.probation_period = 1.
pipeline.persistor.parameters.forget_after = 30.
smoother.parameters.window = 3
```

Recreate components when calibration changes. Changing ORB descriptor layout
(e.g. patch size) requires clearing galleries; changing feature count does not.

The board defaults to the 3x4 letter GridBoard, DICT_4X4_50, IDs 0-11, 25 mm
markers, 37.5 mm gaps, and **96% print scale**. Origin is board center: X right,
Y up, Z out of its face. The implicit 180-degree PnP rotation rotates pixels and
calibration together and is undone before returning the optical-camera pose.
Do not apply legacy virtual-camera pixel-flipping helpers to these transforms.
Nonzero RGB distortion requires an OpenCV-compatible Brown model; the supplied
bag has zero coefficients despite its Inverse Brown model label.

The pose filter averages camera-in-reference positions and proper rotations,
not Euler angles or arbitrary matrix entries. Missing pose, reference changes,
capture gaps over 0.5 s, repeated timestamps and direction changes reset history.
Window 3 adds roughly one frame of lag; it does not reject pose outliers.

Segmentation retains the tested stateless core in `_tabletop_geometry.py`:
sparse RANSAC, voxelized DBSCAN, conservative cluster splitting and metric/ROI
filters. Half-resolution processing uses exact measured rays and restores native
masks/boxes; it needs even image dimensions and rectified zero-distortion depth.
The default footprint limit is 20 cm, excluding height. Narrow wall fragments
can still pass, and larger packages need larger limits. Only one support is
tracked. Initial search assumes its normal mostly follows image Y; adjust
`support_normal_y_min` for substantially different camera orientations.

Plane memory uses ordinary rigid transforms into the supplied reference. Fresh
depth validates/refits it each frame. Poor support, excessive tilt or RMS above
4 mm triggers reacquisition. Without pose, fresh depth-only segmentation still
runs. Missing support/depth returns empty results with error diagnostics.

Appearance uses calibrated RGB ownership with a depth z-buffer (including
non-object depth), cropped convex silhouettes, interior-only ORB patches and
color histograms. Convex completion assumes boxes/cylinders. Default depth
projection stride is 4 and ownership-mask scale .5; native RGB crops supply the
descriptors. Comparison scores are heuristics, not probabilities.

Persistence uses one-to-one matching with 12 cm ordinary position gating and
up to 80 cm recovery requiring distinctive appearance and mutual ambiguity
margins. New IDs need one second of continuous probation. Confirmed missing
tracks survive 30 capture seconds. Galleries retain the original trusted view
plus at most two later views. Suspected hand/object merges can remain unassigned.
Identical packages and major view changes remain ambiguous. Snapshots copy
positions and share immutable signature arrays.

## Classification behavior

`product_setup.py` holds canonical products and legitimate aliases, including
Supramycina. OCR errors are handled with fuzzy matching rather than mistake aliases.
The reader tests both perpendicular text axes and both reading directions;
perspective-rectified lines handle tilt. RapidOCR 1.x is pinned because the fast
reader uses its component APIs. Model thread count is initialization-only;
recognition settings can change between calls/jobs. The reader owns no executor.

Acceptance requires text similarity >= .75, a lead over the next product >= .15,
and **OCR confidence >= .60**. Raw OCR retains readings from .30 for inspection.
Multiple accepted product names in a crop are ambiguous. These confidence and
similarity scores are not calibrated probabilities.

Remembered identity **can change**: stronger unambiguous evidence replaces it,
ranked by similarity, then OCR confidence, then match margin. Equal, weaker,
unreadable or ambiguous readings preserve the best label and its source timestamp.
Labels survive temporary occlusion and disappear when the track expires. Product
types do not merge or reassign object instances. Conflicts freeze gallery learning
through a generic ambiguity signal, not product-specific logic inside detection.

`ObjectClassifier` loads the model in one worker and permits one outstanding
job, with no backlog. Refresh uses five capture seconds. Each job copies RGB
and retains source boxes, IDs and reference positions. Polling accepts results
only for retained confirmed IDs; moves over 8 cm require visual verification
since submission. Clear/reset generations reject stale results. Workers extract
text; matching and label updates occur in the owner thread. OCR still consumes
CPU, even though detection does not wait for inference.

The classification demo defaults to zero probation so a single paused frame can
be classified. Use `--probation 1` for normal acceptance. Headless runs drain the
worker after playback, with `--wait-seconds` (default 60) as the timeout.

## Component demos and saved evidence

Each public component has a main-guard DB3 demo. Later stages reuse the frontend
in `detection_demo.py`:

```powershell
.venv\Scripts\python.exe appearance.py --start-frame 137 --object 0 --reference-object 0
.venv\Scripts\python.exe object_persistor.py --start-frame 137
.venv\Scripts\python.exe detection_pipeline.py --start-frame 137 --features 120
.venv\Scripts\python.exe ocr_reader.py --start-frame 137
.venv\Scripts\python.exe product_matcher.py --start-frame 137
.venv\Scripts\python.exe classification_memory.py --start-frame 137
.venv\Scripts\python.exe classification.py --start-frame 137 --probation 1
```

Appearance comparison uses the first processed frame as a demo reference; the
extractor itself remembers nothing. JSON includes comparison scores, features,
association events and lifecycle diagnostics. Only ID/product labels are drawn
on current frames; old OCR polygons are never painted on newer frames.

```powershell
$env:OMP_NUM_THREADS='4'
.venv\Scripts\python.exe classification.py --headless --start-frame 137 --frames 60 --probation 1 --output outputs/complete_pipeline
.venv\Scripts\python.exe -m unittest discover -v
```

`--output` saves `measurements.json` and `last_frame.png`. Segmentation also saves
configuration/native masks. Classification saves timestamped crops, raw readings,
source anchors and accepted/rejected IDs. Outputs are ignored by Git. Use
`--frames 1` to inspect a particular image.

Local frames 137-196 retained exactly five IDs, keeping the final off-screen
object missing. OCR identified Melatonina, Ibuprofeno and Colgate; two candidates
remained unknown. The background batch took about 2.2 s. With 180 features,
detection alone measured 109 ms median / 135 ms p95, excluding pose estimation,
bag reads and rendering. `--features 120` retained the same five IDs at 99 ms
median / 114 ms p95. Full demo processing is slower; a complete 10 FPS live
system is not established by these measurements.
