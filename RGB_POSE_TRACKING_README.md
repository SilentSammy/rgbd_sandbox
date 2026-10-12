# RGB board pose tracking

`board_pose_setup.make_estimator()` now returns `BoardPoseTracker` by default.
The board-pose, point-memory, and object-persistence demos all use it.
For an unfiltered comparison, call `make_estimator(camera, tracking=False)`.
The existing board axes, implicit 180-degree convention, and 96% print scale
are unchanged. Depth is not used for pose estimation.

```powershell
python object_persistence_demo.py --start-frame 137
python board_pose_demo.py recordings/20261005_194727.db3
python -m unittest discover -v
```

## How it works

The supplied puzzlebot `tracking.py` and `marker_det.py` provided the reference
for rotation-safe smoothing and marker-feature tracking. This implementation
is self-contained in `board_pose_tracking.py`; it does not import that project.

- Average camera position and orientation over the latest three valid poses,
  rather than a wall-clock EMA. Orientations use a proper rotation mean, not
  component-wise Euler-angle or matrix averaging. Under uniform motion, this
  introduces about one source-frame of lag (roughly 33–38 ms in these bags).
- Decode all board markers on each visited frame. Fresh detections take priority.
- For missing detections, track features inside each previously decoded marker
  using batched forward/backward LK and a separate anchored homography per ID.
  Flow consistency, RANSAC inliers, quad geometry, and marker-patch similarity
  must pass. Features on neighboring markers cannot resurrect a hidden marker.
- Fallback expires after three consecutive visited frames without re-decoding
  that marker. At least two decoded/tracked markers are still required for PnP.
- Reset features and filtering on jumps, repeated frame indices, or playback
  reversal. A missing pose clears the filter and returns `None`, never a stale
  pose. Pausing does not add samples. Object memory is independent of these resets.

The returned transform and PnP annotation/projection result both use the filtered
pose. `tracker.raw_pose` exposes the current unsmoothed PnP result, including
any accepted tracked corners. Pass `frame_index=` when navigating recordings.

## Recording checks

Full sequences were processed in order. Step statistics below use only adjacent
pairs for which both methods returned poses. These measure jitter/spikes, not
absolute accuracy: the camera moves and there is no ground-truth trajectory.

| Recording | Raw poses / total | Tracked poses / total | Largest orientation step, raw → filtered | Pose time median |
| --- | --- | --- | --- | --- |
| `20261005_194727` | 358 / 361 | 361 / 361 | 10.30° → 1.64° | 12.7 ms |
| `20261005_195417` | 310 / 315 | 315 / 315 | 26.25° → 1.80° | 17.5 ms |

Around the reported failure, frames 145 down to 125 and back to 145 were freshly
segmented (41 visits). With the initial five IDs already accepted, the raw pose
created 11 IDs and changed associations on 15 visits. The new pose kept all five
IDs without an association change. The 12 cm matching gate was unchanged.

Per-frame results are in `exports/*_rgb_tracking.json` and
`exports/rgb_pose_tracking_problem_clip.json`. Unit tests cover averaging,
rotation wraparound, per-ID fallback, covered-marker rejection, fallback expiry,
navigation resets, and transform/PnP agreement.

This is short-term tracking, not a guarantee under occlusion or rapid motion.
Strong perspective changes or blur can fail its conservative checks; complete
board loss still means no pose. Smoothing can also delay real sudden movement.
