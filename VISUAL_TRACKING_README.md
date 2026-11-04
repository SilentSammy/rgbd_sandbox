# Appearance-assisted object memory

The live OCR demo now uses `AppearanceMemory`, with `RGBObjectAppearance`
configured through `ObjectPersistor(..., appearance=True)`. SAM remains off.
`PerceptionPipeline` also enables this for its default persistor. Explicit
depth-only persistors and the original `ObjectMemory` remain available.

For each segmented candidate, calibrated half-resolution depth-to-RGB
correspondence selects the nearest measured surface. A convex silhouette fills
depth holes across printed faces (appropriate for these boxes/cylinders).
A cropped RGB silhouette is resized to at most 240 pixels on its
longest side. The mask is eroded enough that ORB descriptor patches lie inside
the object, rather than on the wall/table outside it. Each signature stores a
48-bin hue/saturation histogram and up to 180 ORB descriptors. Feature matches
use a ratio test, unique correspondences, affine RANSAC, an inlier minimum and
spatial coverage. These scores are association heuristics, not probabilities.
Weak partial-view matches with enough inliers also receive a reverse-direction
check, without relaxing the geometric acceptance thresholds.

Ordinary nearby detections use position and appearance in one-to-one Hungarian
assignment. Recovery after a gap or a move beyond the usual 12 cm gate requires
strong appearance and a margin over competing tracks AND competing detections.
Recovery is bounded to 80 cm in board coordinates. Color-only distant recovery
is limited to feature-poor signatures with exceptionally similar color and a
clear margin. Richly textured white packages cannot recover on color alone.
Identical packages remain ambiguous rather than receiving arbitrary old IDs.

A candidate substantially larger than a nearby confirmed object, or a larger
candidate still containing its strong features, is treated as suspicious.
It can remain unassigned instead of moving the old ID onto a hand/box merge or
creating a second persistent identity for that merged region. This is not a
hand detector: unrecognized hand fragments can still appear as candidates.
Gallery updates require stable position, compatible size, strong feature/color
agreement and no OCR-label conflict. The first view is preserved, with at most
two additional trusted views. No templates are updated while missing.

Confirmed missing tracks retain their IDs, templates and remembered OCR labels
for 30 seconds of accumulated capture time (`--forget-after`). Delayed OCR can
attach after a move greater than 8 cm only if strong visual correspondence has
verified that same ID since the OCR job's capture. Known product labels are
also stored on the tracks; conflicting OCR freezes gallery updates. A medicine
name alone does not merge two physical instances or resurrect an expired ID.

Run as before on the robot:

```bash
cd ~/stretch_workspace/rgbd_sandbox/medicine_segmentation_compare
bash run_live_ocr.sh
```

Use `--no-appearance` for the position-only comparison; use `--forget-after 5`
as well to reproduce the former retention period. C clears memory. The label
panel's `[R1]`, `[R2]`, etc. counts recoveries. `events.jsonl` records recovery
distance, score and inliers, plus suspicious candidate indices. `latest.json`
and G snapshots include track visibility, stored label, gallery count,
association method and appearance-extraction time. Timers use frame dt.

Tests:

```powershell
.venv\Scripts\python.exe -m unittest test_rgbd_visual_memory test_rgbd_object_persistence test_rgbd_object_ocr
```

Synthetic tests use actual ORB descriptors and cover 45 cm recovery, position
swaps, ambiguous identical boxes, different text on similarly colored boxes,
merge rejection, frozen/bounded galleries, expiry, background exclusion and
delayed OCR after verified movement. They do not prove physical hand-occlusion
performance; that requires the live manipulation test. Major viewpoint changes,
blur, identical packaging or too few visible features may leave a new candidate
unmatched. Longer retention and conservative association do not repair a bad
segmentation or guarantee object identity.

Verified November 4, 2026: all 96 root tests and the 44 relevant tests on the
robot passed. A 60-frame moving-camera bag replay retained the original IDs
for its three recognized medicine boxes. A controlled probe using real RGB-D
signatures from captured frames 0 and 59, with simulated two-second occlusion
and 45 cm displacement, recovered five of six objects from one initial view;
one remained unconfident. This is not a physical hand-manipulation benchmark.
Robot evidence: `medicine_segmentation_compare/outputs/flicker/live_check/appearance_probe.json`.
The final live run is `outputs/ocr/20261104_093710_041431/` in the experiment
folder. Its sampled median loop time after startup was 81.2 ms, core 66.7 ms,
and RGB signature extraction 27.4 ms. Actual manipulation, scene complexity,
OCR load and retained-track count can change these timings.
