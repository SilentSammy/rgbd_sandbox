# Asynchronous live refinement

Run from `medicine_segmentation_compare`, using its SAM-enabled environment:

```bash
env -u PYTHONPATH USE_TF=0 .venv/bin/python live_refinement_demo.py \
  --model models/sam2.1-hiera-tiny --max-dimension 600
```

For the current close-range robot scene, launch with `--scene-depth 1.1
--plane-min-inliers 300`. On the inspected live depth capture, the default
plane search chose a lower surface and produced zero objects. These settings
recovered the tabletop and seven proposals in about 54 ms (six packages plus
an edge/background proposal). They change this demo's segmentation settings,
not the shared baseline defaults. Increase scene depth for more distant objects.

The left window shows current depth proposals and persistent IDs in yellow.
The table plane is anchored to the board and checked/refitted against current
depth; see `../TABLE_PLANE_README.md`. T clears it for relearning.
Depth proposals are drawn directly from segmentation, even without board pose
or tracks; unassociated proposals display their frame-local candidate label.
The right shows delayed SAM geometry reprojected into the current frame in
green, falling back to yellow until usable refinement exists. Green labels
include capture age in seconds. C clears IDs and geometry; G saves a debug
snapshot; Q/Escape exits. Linux SIGUSR1 also requests a snapshot.

Detection/pose/persistence run in the main process. A separate spawned process
loads SAM once and handles one batch of confirmed-object boxes at a time.
Only an idle worker accepts a new capture: no growing queue of stale images.
CPU inference defaults to two threads; `--sam-threads` changes this. The
two-second `--refresh-seconds` interval is a minimum between captures, not a
guarantee of inference speed. Model loading and SAM failure leave coarse
detection running.

Each request retains synchronized RGB/depth, capture timestamp, filtered board
pose, persistent IDs, coarse reference anchors, and depth ranges. SAM receives
RGB and boxes. Returned masks select actual capture-time depth measurements
through the calibrated, nearest-surface RGB/depth correspondence. Seed depth
ranges exclude much of the background/table contamination. Selected measured
points are unprojected with the existing board pixel convention, transformed
into board coordinates, and retained per ID. Each display frame projects those
fixed points with its current filtered pose and bounds their projection.

These are visible-surface points, not inferred cuboid corners or a complete
object model. Novel viewpoints can expose unmeasured surfaces. Geometry stays
fixed in board coordinates; it is not translated to follow a moving object.
The demo hides refinements without a current pose and confirmed visible ID,
after 15 seconds (`--max-age`), after more than 8 cm of coarse-anchor shift,
or when current coarse depth/box checks disagree. These checks are rough
consistency gates, not full collision/occlusion or grasp validation. A newly
captured SAM result can replace stale geometry. Current ID association and
board-pose noise still limit correctness.

`outputs/refinement/<run>/` contains configuration/calibration, periodic timing
and acceptance summaries in `events.jsonl`/`latest.json`, and a folder per SAM
request: original RGB, SAM overlay, original depth and masks, captured pose,
accepted board-coordinate points, prompts, IDs, scores and timings. G saves the
current raw RGB/depth, both annotated views, stored geometry, current pose and
rejection reasons. Snapshot disk writes may briefly pause the main loop.

Offline smoke runs also accept `--bag ../recordings/20261005_195417.db3
--start-frame 137 --headless --frames 100`. Bags process as fast as possible;
short runs can end before the background model finishes loading. There is no
robot movement/control in this demo.
