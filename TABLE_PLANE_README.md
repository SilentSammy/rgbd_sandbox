# Board-anchored tabletop tracking

`rgbd_table_plane.BoardTabletopSegmenter` stores one support plane and its
measured footprint center in fixed board coordinates. `PerceptionPipeline`
passes the filtered pose to this segmenter after RGB pose estimation. The
stateless `TabletopSegmenter` remains available for direct depth-only use.

An initial RANSAC result establishes the plane. On subsequent frames, five
points on the reference plane are converted into current depth-camera
coordinates using the same RGB/depth calibration and board pixel convention
as object persistence. They define the predicted camera-frame plane.

Fresh depth samples within 15 mm of that prediction must meet an inlier count
and span a surface rather than a line. A robust local least-squares fit removes
samples more than 10 mm from its plane and rejects orientation changes above
five degrees. This fitted camera-frame plane drives current segmentation;
its board-frame estimate updates the stored plane with a 5% gain. This uses
actual current measurements, not old masks or boxes.

If current depth cannot support the prediction, a full RANSAC search runs.
A newly acquired surface replaces the old reference plane. Missing board pose
also uses a fresh search, preserving coarse detection without applying stale
camera coordinates. The reference plane remains available for validation when
pose returns. Geometry resets clear the reference plane. This continues to
track **one** tabletop, not multiple shelves.

The live SAM experiment uses this segmenter automatically. **T** clears the
plane for relearning; **C** clears object memory/refinements while retaining
the table. The display and JSON diagnostics report table state, support sample
count, RMS residual, and reference/camera plane coefficients. G captures a
snapshot. The standalone live detection demo also uses the tracked plane.

Camera motion is compensated through board pose, but large pose errors,
occlusion, insufficient depth, or a moved reference board can still require
reacquisition. The local refit reduces sensitivity to small pose noise. The
depth sampling follows the existing segmenter's pinhole geometry; the robot's
current depth/color calibration has zero distortion. Point conversions also
honor calibrated RGB distortion and the existing implicit 180-degree board
pixel convention, which is not a plain optical-frame rigid transform.

Checks:

```powershell
.venv\Scripts\python.exe -m unittest test_rgbd_table_plane test_rgbd_perception test_tabletop_optimization
```

Tests cover camera translation/rotation, off-center intrinsics, nontrivial
RGB/depth extrinsics, the 180-degree board convention, distorted point
roundtrips, missing pose/depth, reacquisition/reset and half-resolution output.

Verified November 3, 2026:

- All 74 root tests and five asynchronous-refinement tests passed.
- The captured 90-frame live sequence, replayed with its actual RGB board
  poses, had zero plane failures (one acquisition, 89 tracked frames), versus
  19 failures with the stateless segmenter. Median segmentation times on the
  robot under concurrent load were 21.6 ms tracked versus 33.0 ms stateless.
- Frames 137–196 of `recordings/20261005_195417.db3` had no segmentation errors
  across approximately 18 cm of estimated camera translation and 4.5 degrees
  of rotation. Five frames acquired/reacquired the table and 55 tracked it.
- The restarted live SAM demo's initial 16 periodic samples all had six coarse
  objects and no segmentation errors; median loop time was 51.7 ms. These are
  sampled observations, not a guarantee that every future frame will succeed.

Local replay evidence: `exports/tabletop_flicker/moving_bag_benchmark.json` and
`exports/tabletop_flicker/board_tracking_benchmark.json`. Robot replay evidence:
`medicine_segmentation_compare/outputs/flicker/live_check/board_tracking_benchmark.json`.
