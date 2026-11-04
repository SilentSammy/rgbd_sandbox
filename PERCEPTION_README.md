# Detection phase

The reusable entry point is `PerceptionPipeline` in `rgbd_perception.py`.
It owns processing state, not acquisition, playback, windows, robot control, or
OCR. `object_persistence_demo.py` is now a bag/playback/display adapter.

```text
RGB ── BoardPoseEstimator ── PoseFilter ──┐
                                       ├── ObjectPersistor ── frame result
Depth ── TabletopSegmenter ──────────────┘
```

## Stage contracts

The filtered pose also feeds the default table-plane tracker; see
[TABLE_PLANE_README.md](TABLE_PLANE_README.md).

- `BoardPoseEstimator.estimate(rgb, frame_index=None)` returns an unfiltered
  `(reference_to_camera, pnp)` pose or `None`. It keeps per-marker RGB feature
  tracking. The matrix is 4x4, with translation in meters, in the existing
  board/PnP convention. It does not use depth or filter across poses.
- `PoseFilter.update(pose)` returns a filtered pose, including consistent PnP
  annotations. It averages the latest three processed camera-in-reference
  poses. `None` clears the history and returns `None`, never a stale pose.
- `BoardTabletopSegmenter.segment(depth, pose=filtered_pose)` is the pipeline
  default. It stores one plane in board coordinates, validates/refits its
  current projection against depth, and reacquires it when support fails.
  Geometry resets also reset this plane; missing pose uses fresh depth search.
- `TabletopSegmenter.segment(depth)` is the stateless depth-only alternative.
  Both segmenters return masks, candidate-object geometry,
  padded depth-image rectangles, the support plane, and segmentation diagnostics.
  By default it processes half-resolution depth, then restores native masks and
  pixel coordinates. NumPy voxel membership replaces Open3D's ragged traces;
  plane fitting and DBSCAN are unchanged. Metric geometry comes from coarse
  samples, not from every pixel of the expanded native mask.
  Its object coordinates are in the depth-camera frame. A missing support plane
  raises `RuntimeError`, which the pipeline reports as a failed segmentation.
- `ObjectPersistor.update(objects, pose, dt)` localizes cluster centers in the
  fixed reference and performs gated one-to-one matching, probation, and lost
  track retention. It returns track snapshots with persistent IDs. With no
  pose, it creates no localized observations but still ages memory.

The pipeline's default persistor also receives `frame` and `segmentation` as
keyword arguments for masked RGB appearance. Its `AppearanceMemory` extends
the position-only hooks with conservative instance recovery. See
[VISUAL_TRACKING_README.md](VISUAL_TRACKING_README.md). Explicit
`ObjectPersistor(calibration, memory)` retains the geometry-only baseline;
use `appearance=True` to enable visual signatures for a supplied visual memory.

Each stage can also be used directly. Calibration is configured once; individual
frames carry their capture time, RGB array, and depth array. Frame/calibration
types live in `rgbd_types.py`, independent of bags or ROS. The reader re-exports
them, so existing imports remain valid. `make_estimator()` and `BoardPoseTracker`
still compose estimation/filtering for the older pose demos.

## Core usage

```python
from rgbd_perception import PerceptionPipeline
from rgbd_types import Frame

pipeline = PerceptionPipeline(calibration)

# In the source's capture loop:
frame = Frame(capture_timestamp_ns, rgb_uint8, depth_float_meters)
result = pipeline.process(frame)

for track in result.tracks:
    if track.confirmed and track.visible:
        detection = result.segmentation.objects[track.detection_index]
        # track.id: persistent instance ID; track.position: reference XYZ meters.
        # detection: this frame's depth contour/centroid/visible dimensions.
```

`PerceptionResult` includes the input frame, raw and filtered poses, segmentation,
track snapshots, capture `dt`, status, and timings for pose estimation, filtering,
segmentation, persistence, and total core processing. Track indices refer only
to that result's segmentation; they are not persistent identities. Missing
confirmed tracks have `detection_index=None` and remain in the result.

Track snapshots own their positions, so subsequent processing cannot silently
change previous IDs/positions. Image arrays are borrowed from the supplied
`Frame`: a live adapter must copy SDK buffers if they will be reused while
results/crops are still in use. The later classification phase can associate jobs with this result's
source timestamp and track ID, without introducing OCR into these stages.

## Acquisition, conventions, and timing

Inputs must be synchronized RGB uint8 and floating-point depth **in meters**,
at the calibrated dimensions/orientation. Do SDK depth-unit conversion,
color-order conversion, image rotation, and matching calibration changes in
the source adapter. Recreate the pipeline if calibration changes.

The existing implicit 180-degree board convention and 96% print scale are
preserved. Use `rgbd_geometry.py` for object/image/reference conversion: the
original RGB-to-PnP pixel mapping is part of the convention. Applying a bare
inverse pose directly to depth-camera points would bypass both that mapping
and the depth-to-color extrinsics. Visible extents are not complete cuboids,
and segmentation candidates are not certified collision-free grasp targets.

Live mode requires increasing capture timestamps and computes dt from their
difference, not processing duration or wall-clock time. Skipped captures still
advance timers by the elapsed capture time. Duplicate/out-of-order timestamps
are rejected before changing state. Call `reset()` for a new stream/reference
session; old IDs are not recycled within the pipeline instance.

The bag demo enables `allow_reverse=True`: backward visits use absolute capture
dt and are processed as new observations. Optional `frame_index` resets geometry
on jumps/reversals, without clearing object memory. C clears only object memory.

The core is synchronous and should be called serially; it does not queue frames.
For the robot, acquisition should supply the latest synchronized frame rather
than accumulate a processing backlog. The three-frame filter's delay is about
one **processed** frame under uniform motion; at a low processing rate that is
longer than 33 ms. Per-stage timing will establish whether later independent
stage rates or further segmentation optimization are needed. The robot's live
RealSense source and launcher are described in `LIVE_DETECTION_README.md`.
No robot motion, ROS-node adapter, OCR, or grasp planner is included in this phase.

## Demo and checks

```powershell
python object_persistence_demo.py --start-frame 137
python object_persistence_demo.py --start-frame 137 --segmentation-scale 1
python -m unittest discover -v
```

Playback controls and max-dimension sizing are unchanged. The console now
separates core perception time from display projection/annotation time, so
window rendering cannot be mistaken for algorithm latency.

The default `--segmentation-scale 0.5` scales pixel-count thresholds with image
area and pixel-distance thresholds with image length. Metric thresholds stay
unchanged. Masks remain native-size, invalid native depth pixels remain unlabelled,
and bottom padding retains its original native-pixel value. Scale 1 enables the
NumPy bookkeeping optimization without reducing depth resolution. The standalone
`segment_tabletop_objects()` function defaults to scale 1 for older demos; pass
`processing_scale=.5` explicitly or use the pipeline's default segmenter.

RGB pose estimation always uses the original RGB/calibration. Only the demo's
display projector uses reduced depth and display-sized RGB intrinsics to avoid
building native RGB maps merely for annotations. Playback uses a shorter event
wait while playing; shared keys and pause behavior are unchanged.

Complete sequential runs of `20261005_195417` (315 frames) and `20261005_194727`
(361 frames) had no support-plane or board-pose failures. Excluding the first
frame, core median/p95 times were 50.1/58.5 ms and 45.2/56.2 ms respectively.
These exclude capture, drawing, GUI event handling, and robot execution. The
41-visit problem segment preserved its five IDs; September recording
`20260924_113006`, frame 4, still separated five objects.

Half resolution is an accuracy/performance tradeoff, not exact output parity.
Across the 315-frame object recording, counts differed from native segmentation
on 66 frames. Where counts agreed, left-to-right centroid differences were
3.1 mm median, 7.6 mm p95, and 16.8 mm maximum. These are differences from the
other implementation, not errors against ground truth; equal counts do not prove
identical membership. Use scale 1 to compare difficult scenes. Native-depth
grasp-point refinement has not been implemented by this optimization.

Tests cover the independent stages, filtered-PnP consistency, vectorized geometry
against the original pixel mapping, capture-time probation, pose/plane loss,
frame validation, navigation resets, new sessions, and stable result snapshots.

Before this optimization, on `20261005_195417.db3`, 87 visits (137 through 180 and back to 137) matched
the pre-refactor estimator's poses exactly and retained the original five IDs
through probation. The separate 41-visit problem segment (145 down to 125 and
back to 145, initial tracks accepted immediately) also had zero pose differences
and zero identity changes. These are regression checks, not pose ground truth.
Core median latency was approximately 224–227 ms: 19–20 ms pose, 0.6 ms filter,
205 ms segmentation, and 0.4 ms persistence. This is about 4.4 processed FPS
on this laptop, excluding acquisition/display, not a claim of real-time 30 FPS.
