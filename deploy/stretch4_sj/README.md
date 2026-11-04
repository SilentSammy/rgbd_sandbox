# SJ Python / Stretch 4 RGB-D

One shared environment: `stretch4_rgbd/venv`. The `venv` at the project root and
the old `scripts/venv` path are symlinks to it. The original scripts environment
is retained as `scripts/venv.original`.

For a new terminal:

```bash
cd /home/hello-robot/stretch_clean_ws/sj
source activate.sh
python scripts/boilerplate.py
```

The boilerplate only reads the current lift position through `RobotClient`.
The body server must be running for that script. Activation also removes the
old `/usr/lib/python3/dist-packages` PYTHONPATH override so the camera API gets
the newer OpenCV. ROS overlay paths are preserved. This is the RGB-D project
environment, not a replacement environment for the gamepad tools.

You can bypass activation with the Python launcher:

```bash
./python.sh scripts/boilerplate.py
./python.sh check_setup.py
```

Capture RGB-D calibration data from any working directory:

```bash
/home/hello-robot/stretch_clean_ws/sj/capture_rgbd.sh --camera left --lidar left
```

Press Space in the terminal to capture a view, and Q to finish. Captures are
saved under `stretch4_rgbd/data`. This capture tool does not open a preview
window. It uses existing calibration; capturing does not install new calibration.

Sensor smoke test (ten left RGB-D frames, no robot motion commands):

```bash
timeout --signal=INT --kill-after=10s 60s ./python.sh check_setup.py --live
```

The environment follows upstream's `--system-site-packages` recommendation to
access installed robot drivers. It therefore inherits host packages; it is not
fully isolated. Use the project launchers to select the right interpreter.

## Current calibration blocker

The robot currently has no `calibration_cameras/camera_extrinsics.yaml` under
`/home/hello-robot/stretch_user/stretch-se4-4020`. Live RGB-D requires that file,
including `transform_right_lidar_to_head_center`. The smoke test reports this
before opening hardware. Ask the owner for the factory camera calibration for
**stretch-se4-4020**; do not substitute another robot's calibration. No factory
calibration files were changed and no calibration motion was attempted.

Matplotlib is installed in the project venv. Its `00_sj_mpl_toolkits.pth` restores
normal namespace search order to prevent the system's legacy namespace hook
from selecting an incompatible system `mplot3d` with the newer Matplotlib.
