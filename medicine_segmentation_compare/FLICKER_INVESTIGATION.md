# Live tabletop flicker investigation — November 3, 2026

Captured 90 consecutive synchronized RGB/depth frames (about three seconds)
from the robot's D435i. Live demo resumed after capture. Replayed identical
depth inputs without SAM or board tracking, using instrumentation in a separate
process; production segmentation code was not changed by this investigation.

Robot evidence: `medicine_segmentation_compare/outputs/flicker/live_check/`.
This includes calibration, `sequence.npz`, frame-by-frame cluster/plane traces,
sample RGB/depth images, and `summary.json`. Local copies of the initial
four-way replay are in `exports/tabletop_flicker/`.

| Configuration | Missing-plane frames / 90 | Median segmentation time |
|---|---:|---:|
| Current demo: max depth 1.1 m, minimum plane inliers 300 | 19 | 28.6 ms |
| Lower minimum inliers to 100 | 18 | 30.0 ms |
| Denser plane grid: stride 2 instead of 4 | 17 | 56.0 ms |
| Search 8 candidates instead of 4 | 11 | 31.2 ms |

A final control reused the **first successful plane** across the same short
static sequence: **zero missing-plane/zero-object frames out of 90** (67 frames
had seven accepted clusters, 23 had six). Its median was 22.9 ms; the concurrent
replay's current configuration measured 34.2 ms. See the five-way final results
in `exports/tabletop_flicker/control_summary.json`. This control isolates the
plane search; a live implementation must transform/validate the plane as the
camera moves. No claim of moving-camera correctness follows from freezing it
on a static sequence.

These timings are diagnostic replay timings under concurrent live/SAM load,
not a full-loop benchmark. Half-resolution processing scales the current
plane-inlier requirement from 300 to 75 and object-pixel requirement from
500 to 125.

The main failure is greedy unconstrained plane extraction. Each RANSAC call
fits the dominant plane, rejects it if its orientation is unsuitable, but
still removes all its inliers before searching again. A slanted plane can
therefore consume genuine table points without becoming an accepted support.
The search stops after four candidates.

In failed sequence frame 8, there were 247 samples within 10 mm of the
reference table plane. The successive rejected candidates removed 9, 125,
38 and 4 of those samples, leaving 71. All four fitted plane normals failed
the tabletop-orientation test. On successful frames the corresponding support
count ranged from 235 to 287; on failed frames it ranged from 235 to 267.
This reference-plane check used the normalized median successful plane as a
diagnostic, not a production temporal estimator. The table measurements were
present even when the plane search failed.

Successful fits also varied: predicted table Y at depth-camera X=0, Z=0.75 m
spanned 4.39 mm. With object membership starting 10 mm above the table,
this variation can change the bottom edge of a mask. Depth noise, holes,
voxelization and hard cluster/size gates can add individual-boundary changes.

The six-versus-seven accepted-object count in this capture includes a small
left-edge background fragment crossing the 125-pixel coarse threshold: one
example had 134 samples and passed, the following frame had 101 and failed.
The six remaining accepted clusters occupied the six package positions.
Object counts alone should not be interpreted as missed medicine counts.

Recommended production change: retain an accepted table plane in the fixed
reference frame, transform it into the current depth-camera frame, validate
its support against current depth, and refine it locally. Search globally
only at initialization or when that validation fails. This requires handling
camera motion and the existing board pixel convention consistently; simply
reusing an old camera-frame plane while the camera moves would be wrong.
An orientation-constrained plane search would also avoid destructive peeling
of geometrically unsuitable candidates. Do not merely preserve old image
boxes: they can be wrong after camera/object motion.

The current live demo remains running with its previous settings. This
investigation adds diagnostics, not a temporal-plane fix.
