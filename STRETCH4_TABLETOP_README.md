# Stretch 4: one-shot tabletop segmentation

On the robot, from `~/tec_workspace`:

```bash
python3 stretch4_tabletop_demo.py
```

Keep the robot and objects still during capture. The script warms up the right
camera, takes one RGB-D frame, releases the camera/LiDAR streams, and segments
that frame once. The OpenCV results remain frozen until Q/Esc or a window closes.
No gridboard, camera localization, object persistence, or OCR is used.

RGB and depth are fisheye-undistorted together into an upright pinhole view.
The segmentation module receives the matching rectified intrinsics and depth
in meters. The table is estimated from that depth, not from a known world pose.
Gray indicates the estimated table; colored regions and rectangles indicate
accepted object clusters. Height annotations are distances above the table.

Snapshots, calibration, and PNG results are saved under
`captures/tabletop_<timestamp>/`. Reprocess the same snapshot without opening
the sensors:

```bash
python3 stretch4_tabletop_demo.py --input captures/tabletop_<timestamp>/rectified_rgbd.npz
```

Useful options:

- `--max-depth 2.5`: far depth cutoff, in meters.
- `--max-object-size 0.35`: maximum camera-axis object extent, in meters.
- `--measured-only`: use only actual LiDAR samples for segmentation.
- `--max-dimension 800`: maximum displayed image dimension.
- `--crop-y 350 480`: restrict segmentation to that vertical band in the full
  displayed image. Coordinates are converted to native pixels using
  `--max-dimension`; the cropped principal point is adjusted accordingly.
- `--no-gui`: save results and exit, for headless checks.
- `--output-dir PATH`: choose the output directory.

By default the demo uses the official dense-depth interpolation. Filled depths
are estimates, not additional measurements. A separate measured-depth view is
provided for comparison; interpolation and calibration errors can affect object
boundaries and measurements. This is a segmentation experiment, not a validated
grasp planner.

Make sure a tabletop is clearly visible. The algorithm finds a support plane;
if the view contains mostly floor, it can select the floor instead. Inspect the
gray plane region before interpreting colored clusters as tabletop objects.
The snapshot archive remains full-frame, so the crop can be adjusted on replay.

Required project files on the robot: `stretch4_tabletop_demo.py`,
`rgbd_tabletop_segmentation.py`, and `rgbd_types.py`. Dependencies are OpenCV,
NumPy, Open3D, the robot SDK, and `stretch4_emulated_rgbd`. The bag reader and
board-pose modules are not needed. The existing camera viewer is unchanged.
