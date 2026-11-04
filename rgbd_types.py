"""Source-independent RGB-D data: RGB uint8, depth float meters, capture time ns.

Images and calibration must have the same orientation and native dimensions.
Camera axes are X right, Y down, Z forward; depth_to_color uses meters.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    distortion_model: str


@dataclass
class Calibration:
    color: Camera
    depth: Camera
    depth_to_color: np.ndarray


@dataclass
class Frame:
    timestamp_ns: int
    color: np.ndarray
    depth: np.ndarray


def rotate_camera_clockwise(camera):
    """Rotate pixels and optical axes clockwise, including Brown distortion."""
    K, D = camera.K, camera.D.copy()
    rotated_K = np.array([[K[1, 1], 0, camera.height-1-K[1, 2]],
                          [0, K[0, 0], K[0, 2]], [0, 0, 1]])
    if "brown conrady" in camera.distortion_model.lower() and len(D) >= 4:
        D[2], D[3] = D[3], -D[2]
    return Camera(camera.height, camera.width, rotated_K, D, camera.distortion_model)


def rotate_calibration(calibration, rotations):
    color, depth, transform = calibration.color, calibration.depth, calibration.depth_to_color.copy()
    axes = np.eye(4)
    axes[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    for _ in range(rotations):
        color, depth = rotate_camera_clockwise(color), rotate_camera_clockwise(depth)
        transform = axes @ transform @ axes.T
    return Calibration(color, depth, transform)
