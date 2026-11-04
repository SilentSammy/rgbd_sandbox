"""Check project imports; --live reads ten RGB-D frames without motion commands."""

import argparse
import importlib
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--live', action='store_true')
args = parser.parse_args()

print('Python:', sys.executable, flush=True)
for name in ('numpy', 'cv2', 'scipy', 'yaml', 'cma', 'zmq', 'rerun',
             'stretch4_body', 'stretch4_emulated_rgbd.api'):
    module = importlib.import_module(name)
    print(f'OK {name}: {getattr(module, "__version__", "")} {module.__file__}', flush=True)

import cv2
import numpy as np
from stretch4_body.robot.robot_client import RobotClient

assert int(np.__version__.split('.')[0]) >= 2, 'RGB-D requires NumPy >= 2'
assert hasattr(cv2.aruco, 'DICT_ARUCO_MIP_36h12'), 'OpenCV is too old'
print('OK RobotClient and required ArUco dictionary', flush=True)
from mpl_toolkits.mplot3d import Axes3D
print('OK Matplotlib 3D namespace', flush=True)

if args.live:
    from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream

    fleet = Path(os.environ['HELLO_FLEET_PATH']) / os.environ['HELLO_FLEET_ID']
    extrinsics = fleet / 'calibration_cameras/camera_extrinsics.yaml'
    if not extrinsics.is_file():
        parser.exit(2, f'Live RGB-D blocked: missing factory calibration {extrinsics}\n'
                    'Restore this robot\'s calibration from its owner before capture.\n')

    streamer = generator = None
    try:
        streamer, generator = get_emulated_rgbd_stream(
            use_left=True, use_left_lidar=True, use_right_lidar=False)
        count = 0
        for frame in generator:
            if frame is None:
                continue
            depth = frame.depth_image
            valid = np.isfinite(depth) & (depth > 0)
            print(f'Frame {count}: RGB {frame.image.shape}, depth {depth.shape}, '
                  f'{valid.sum()} valid depth pixels', flush=True)
            if not valid.any():
                raise RuntimeError('No measured depth in the received frame')
            count += 1
            if count == 10:
                break
        if count != 10:
            raise RuntimeError(f'Stream ended after {count} frames')
    finally:
        if generator is not None:
            generator.close()
        if streamer is not None:
            streamer.stop()
