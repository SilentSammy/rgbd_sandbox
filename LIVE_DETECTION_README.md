# Live detection on Stretch

Location: `/home/hello-robot/stretch_workspace/rgbd_sandbox`.
`cam_test.py` remains the original D405 depth-filter comparison.

```bash
cd /home/hello-robot/stretch_workspace/rgbd_sandbox
./run_live_detection.sh
```

The default selects the unique D435 camera, captures RGB8 and native Z16 depth
at 848x480 / 30 FPS, rotates both images and calibration clockwise, converts
depth to meters, and uses the same detection pipeline as bag playback.
The SDK queue holds one synchronized frameset, replacing unread frames rather
than growing a backlog. Capture timestamps determine probation and missing time.
SDK buffers are copied before reuse. Spatial/temporal/hole-filling filters are
not applied by default because they change the depth evidence at object edges.

C clears persistent memory; G writes a timestamped debug snapshot; Q or Escape exits. Closing either window exits.
`--serial` selects another camera explicitly; `--rotations 0` shows native
orientation; `--segmentation-scale 1` compares native-depth segmentation.
Without the letter board in view, candidate rectangles are still displayed,
but IDs cannot be localized/confirmed. The board retains the existing 180-degree
pose convention and 96% print scaling.

```bash
./run_live_detection.sh --headless --frames 100 --output-dir live_check
```

The output directory contains last-frame RGB/depth annotations and a JSON timing
report (last 600 measurements, keeping continuous runs bounded). Console logs separate core processing from the complete capture/display
loop and report skipped depth frame numbers and RGB/depth timestamp separation.
No robot motion commands are issued.

Live debug snapshots default to `debug_snapshots/`. Each snapshot contains raw
RGB, floating-point depth in meters, annotated views, calibration, raw/filtered
pose, support plane, complete object/track geometry, timings, and the preceding
15 processed RGB-D frames in `history.npz`. State and IDs continue uninterrupted.
The startup line prints the process-specific SSH command for requesting the same
snapshot with `SIGUSR1` when the GUI is inaccessible.

Unexpected exceptions save a crash snapshot and traceback. The launcher waits
two seconds and restarts the process; normal Q/Escape exits do not restart it.
Loss of the GridBoard or support plane remains an ordinary status result and does
not invoke crash recovery.

To show windows on the robot's logged-in desktop from SSH:

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority ./run_live_detection.sh
```

The desktop display/auth path depends on the login session. A normal desktop
terminal sets these automatically. SSH itself does not forward these windows
to Windows; use the robot desktop/remote desktop for interactive viewing.

Recreate the isolated environment with:

```bash
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/python -m pip install --find-links .wheels -r requirements-detection.txt
env -u PYTHONPATH .venv/bin/python -m pip check
```

The launcher clears the ROS shell's `PYTHONPATH` for this process so ROS package
overlays cannot inject unrelated dependencies into the isolated venv.

The robot has a persistent tmux session named `rgbd`, opened in this directory
with the venv activated. Reattach with:

```bash
ssh -t -o ServerAliveInterval=15 hello-robot@192.168.137.64 'tmux attach -t rgbd'
```

Ctrl+B, then D detaches without closing the shell. Keepalives detect a failed
connection; tmux preserves the remote shell if the SSH connection drops.
The D405 can be tested explicitly with `--serial 128422271445 --fps 10`;
848x480 RGB on that device does not support the default 30 FPS.

## Verified on the robot

The isolated environment passes `pip check`, imports all detection dependencies,
and passes 46 detection/pose/playback regression tests. Open3D 0.19 is installed
in the venv; the robot's global packages remain independent of this environment.

On 40 consecutive bag frames (137-176 of `20261005_195417.db3`), every frame had
a board pose, frame 137 separated five objects, and the final four visible
objects had confirmed IDs. Core median/p95 were 85.9/92.1 ms, excluding bag I/O
and drawing.

A 60-frame D435i live GUI run, excluding five startup measurements, had core
median/p95 72.3/79.1 ms and complete-loop median/p95 84.5/92.6 ms, averaging 11.7
processed FPS. Capture remained at 30 FPS, with skipped frames rather than an
accumulating queue. These live figures describe the current wall/floor view,
which has no board pose; they do not prove that full live board tracking and
object association meet the same frame rate. Aim the camera at the letter board
and tabletop to validate that scene. Validation images/timings are in
`venv_display_check/` and `venv_live_check/` on the robot.
