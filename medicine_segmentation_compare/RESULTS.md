# Initial checks, November 2, 2026

All results below use the existing `20261005_195417.db3`, clockwise rotation,
SAM 2.1 Tiny, CPU PyTorch 2.7.1, Transformers 4.57.3, and the actual current
depth segmentation with its 1.05 m centroid cutoff. No models ran on the robot.

Model revision: `de431c4043854a71d8101e17995dfe596bf101a5`.
Verified checkpoint SHA-256:
`48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`.
Model loading reported zero missing, unexpected, or mismatched weights. The Hub
currently labels the checkpoint `sam2_video`; Transformers warns when loading
the image wrapper. The actual image weights loaded completely and inference
was verified. Loading diagnostics are saved in each run's environment/report.

## Frame 137, five depth-box prompts

[Gallery](outputs/20261102_174238_263713/index.html) ·
[Side-by-side image](outputs/20261102_174238_263713/frame_000137_visit_000000/comparison.png) ·
[Full measurements](outputs/20261102_174238_263713/frame_000137_visit_000000/report.json)

Both methods produced five regions. SAM visibly fills depth sampling holes and
produces much more complete silhouettes. It still excluded part of the coffee
jar's glass base despite predicting high mask quality. This illustrates why
predicted quality cannot establish full-object correctness.

The first SAM pass took 5.45 seconds including preprocessing/postprocessing;
depth segmentation took 70.5 ms. These measurements exclude model loading and
artifact export. Image encoder work dominates this CPU experiment; this setup
does not meet the robot's real-time target.

## Upper shelf, manual prompt

[Gallery](outputs/20261102_174411_238258/index.html) ·
[SAM mask overlay](outputs/20261102_174411_238258/frame_000137_visit_000000/sam.png) ·
[Measurements](outputs/20261102_174411_238258/frame_000137_visit_000000/report.json)

The independent RGB point `(170,550)` selected the upper-shelf package, outside
the depth baseline's five proposals. SAM produced a plausible box silhouette.
Measured median depth was 0.85 m. No new support plane or depth proposal was
needed to prompt this mask. SAM time: 4.43 seconds. This tests prompted outlining,
not automatic discovery or drug identification.

## Three-frame image tracking alternative

[Gallery](outputs/20261102_174505_401861/index.html) ·
[Timing summary](outputs/20261102_174505_401861/summary.jsonl)

Frames 137-139 returned five masks each. The first frame used depth boxes; the
next two used previous SAM-mask boxes. SAM times were 5.08, 4.22, and 4.20 seconds;
depth times were 55.1, 45.2, and 31.9 ms. Export separately took 0.57-1.02 seconds
per frame. This short test does not establish identity stability or occlusion
recovery and does not use SAM's video-memory model.

Additional validation: six comparison tests pass, dependency checking is clean,
the GUI ran with 600-pixel windows, and exported RGB/depth/calibration replayed
successfully. Tests cover occlusion-aware depth sampling, mask selection,
empty-support-plane behavior, overlap preservation, prompt labels, and data
roundtrip. Existing robot perception modules and the running live demo were
not changed.
