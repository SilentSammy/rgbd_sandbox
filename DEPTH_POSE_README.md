# Depth-assisted board pose experiment

No changes were made to the existing board estimator, setup, coordinate
convention, or object-persistence demo. This is an independent, unsmoothed test.

Reproduce the full-board recording comparison:

```powershell
python board_depth_pose_experiment.py --plane-scale 2 --max-plane-rms 0.015
```

Default recording: `recordings/20261005_194727.db3`, all 361 paired frames.
Results: `exports/20261005_194727_depth_pose_s2_r0.015.json`.

## Method

1. Detect the known board in RGB, using the existing 96% print scale.
2. Compare current corner detection with OpenCV subpixel refinement.
3. Project measured depth samples into RGB using recorded calibration/extrinsics.
4. Select samples inside an eroded board hull, optionally expanded around it.
5. Fit a consensus plane, then trim outliers and refine by least squares.
6. Intersect RGB corner rays with that plane and rigidly align the known board
   geometry to those 3D intersections. Depth constrains tilt/distance, while RGB
   fixes the board's identity, position within the plane, and in-plane rotation.

The existing estimator's reflected-pixel/PnP convention is preserved. Synthetic
tests check that convention, rigid alignment, and rejection of foreground points.
There is no temporal smoothing, previous-pose initialization, or pose caching.

`--plane-scale 1` uses only the detected board region. A larger patch assumes
the board and surrounding wall are coplanar. This is scene-dependent: it is not
appropriate for a freestanding board, bent paper, or a differently oriented wall.
RANSAC removes shelf-edge/foreground samples, but cannot guarantee the selected
plane is the board surface. Additional geometric consistency checks are needed
before using this as an automatic production estimator.

## Unobstructed recording results

All three methods returned poses on the same 358 frames; three RGB detections
failed. The following compares exactly the same 355 adjacent valid frame pairs.

| Method | Median angular change | 95th percentile | Maximum |
|---|---:|---:|---:|
| Current RGB | 1.10 degrees | 3.86 degrees | 10.30 degrees |
| RGB with subpixel corners | 1.07 degrees | 4.34 degrees | 12.72 degrees |
| RGB + depth, 2x patch, 15 mm residual limit | 0.82 degrees | 3.36 degrees | 6.82 degrees |

Median RGB reprojection RMS was 0.89 px (current), 0.87 px (subpixel), and
0.99 px (depth-assisted). Lower pixel error alone did not predict fewer pose jumps.
The depth stage added a median 34.4 ms, excluding RGB detection and file reading.
Depth-plane RMS ranged up to 13.2 mm in this run, so the permissive 15 mm setting
should not be interpreted as strong depth confidence on every frame.

With a stricter 8 mm residual limit, depth was accepted on 281/361 frames. On
the common 264 adjacent frame pairs, current RGB had p95/max changes of
3.21/10.30 degrees versus 1.97/3.22 degrees for depth. These are subset results;
the rejected frames must not be ignored when evaluating an eventual fallback.

Other trials were less successful. Subpixel refinement alone did not eliminate
jumps. A small board-only depth patch was noisy; a 1.5x patch still produced a
10.58-degree maximum change in this recording. ROI size needs explicit validation.

The camera is moving and there is no ground-truth pose. Angular changes include
real motion: these results demonstrate reduced spikes, not absolute pose accuracy.
The JSON files retain timestamps and per-frame estimates for further analysis.

## Previously failing tracking segment

```powershell
python board_depth_pose_experiment.py recordings/20261005_195417.db3 --start-frame 125 --stop-frame 146 --plane-scale 1.5 --max-plane-rms 0.008
```

All 21 frames accepted depth in this smaller-patch trial. The maximum angular
change fell from 18.01 degrees (current RGB) to 4.96 degrees (depth-assisted).
At frame 132, same-object board-center changes decreased from 120-171 mm to
24-40 mm; at frame 131, from 143-203 mm to 14-24 mm.

A backward association check over frames 145 through 125, starting with the five
tracks already confirmed, used the same 12 cm matching gate for both methods.
RGB created nine IDs and changed identities on eight visits. Depth preserved
the original five IDs on every visit; its maximum same-object position step
over the segment was 62.9 mm, versus 202.7 mm with RGB.
This is a short targeted check, not proof of general tracking correctness.

Expanding to 2x on this partially obstructed recording accepted only 11/21 frames
with the 8 mm limit. The best ROI settings did not transfer unchanged between bags.

## Next step

Combine RGB and depth with explicit quality checks: plane support/coverage,
depth residual, RGB reprojection residual, and consistency of plane distance
with the board pose. Use a smaller ROI or reject foreground contamination when
needed. If depth is unreliable, flag the pose as weak rather than silently
treating an RGB solution as validated. Optimize depth projection to the board
region before targeting a tight real-time latency budget.

Run `python -m unittest test_board_depth_pose_experiment` for the synthetic tests.
OpenCV references: [corner refinement](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
and [pose estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
