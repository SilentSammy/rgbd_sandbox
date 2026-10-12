# Object persistence demo

```powershell
python object_persistence_demo.py --start-frame 137
```

The default bag is `recordings/20261005_195417.db3`. Controls are shared with
the RGB-D viewer: A reverse, S pause, D play, Q/E previous/next, W restart at
`--start-frame`, Escape quit. C clears memory without reusing old IDs.

Each visited frame is segmented afresh. Depth-cluster centers are expressed
in the fixed letter GridBoard frame using the existing camera-pose convention.
Gated, one-to-one minimum-distance assignment associates detections with tracks;
segmentation labels are frame-local and are never used as object identities.

Camera pose is RGB-only: decoded board markers take priority, with short
per-marker optical-flow fallback and a three-frame camera-pose average.
Depth still supplies segmentation and object centers, not camera pose.
Frame jumps and playback reversals reset pose filtering/tracking, not object
memory. See [RGB_POSE_TRACKING_README.md](RGB_POSE_TRACKING_README.md).

Yellow `ID ... ?` rectangles show candidate IDs and probation progress. Candidates
must remain detected for `--probation-period` seconds (default 1); any missed
detection drops them. Colored rectangles indicate confirmed tracks. Missing
confirmed objects retain their IDs for `--forget-after` seconds (default 5);
crosses show their last measured centers projected into the current RGB image.
These are remembered positions, not current detections or proof of visibility.

Lifecycle time advances by `abs(current_timestamp - previous_timestamp)` in
seconds, with zero dt on the first frame. Pausing does not age tracks. Backward
navigation is treated as new observations, not a rewind of memory. Restarting
also uses the timestamp gap: a jump of 5 seconds or more expires existing tracks
with the defaults. Only C explicitly clears memory. Skipped time advances timers
but is not evidence that every intervening frame contained the object.

Without a board pose, observations cannot be localized: probationary candidates
are discarded and confirmed memory ages normally. No camera-coordinate fallback
or reuse of stale poses is performed. Segmentation failures behave the same way.

`--match-distance` sets the 3D matching gate in meters (default 0.12). The default
accommodates observed board-pose jumps of up to approximately 11.5 cm in the
benchmark sequence; it is not a guarantee of correct association in other scenes.
`--max-dimension` controls display sizing. The tracker assumes stationary objects;
pose jitter, large viewpoint changes, changing visible centroids, touching/merged
objects, and close neighbors can still cause mistaken associations or new IDs.
This is geometry-only identity, not medicine recognition or grasp estimation.

OCR is deliberately not included yet. A future background job should carry the
source frame and track ID with its crop. Its result must update that ID, never
the current frame's segmentation index. Discard results whose track has expired
or was cleared; IDs are not recycled, so a stale result cannot label a new track.

Run `python -m unittest test_rgbd_object_persistence test_rgbd_playback` for
deterministic lifecycle, matching, and playback tests.

Checked frames 137 through 180 and back to 137 (87 freshly processed visits):
the 12 cm gate kept the original five IDs, including the rightmost object
leaving the segmentation region and returning. All five passed probation.
The same-object RGB/board projection round trip at frame 137 was accurate to
approximately `1e-13` pixels; that checks conventions, not cross-frame pose accuracy.
