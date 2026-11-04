"""Reference/image conversions preserving the board estimator's pixel convention.

A pose is (reference_to_camera, pnp). Its matrix uses the existing board/PnP
axes; original RGB pixels must go through pnp.to_image_points(), not a bare
inverse transform. All metric positions are in meters.
"""

import cv2
import numpy as np


def image_point_to_reference(pixel, color_z, pose, camera):
    transform, pnp = pose
    pixel = pnp.to_image_points([pixel]).astype(np.float64).reshape(1, 1, 2)
    xy = cv2.undistortPoints(pixel, camera.K, camera.D).reshape(2)
    point = np.array([xy[0]*color_z, xy[1]*color_z, color_z, 1])
    return (np.linalg.inv(transform) @ point)[:3]


def reference_points_to_image(points, pose, camera):
    transform, pnp = pose
    points = np.asarray(points).reshape(-1, 3)
    if not len(points):
        return np.empty((0, 2)), np.empty(0)
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    pixels, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), camera.K, camera.D)
    return pnp.to_image_points(pixels), camera_points[:, 2]


def object_positions_in_reference(objects, calibration, pose):
    """Depth-cluster centers -> RGB pixels -> existing PnP camera -> reference."""
    points = np.array([obj.centroid_xyz for obj in objects], dtype=float).reshape(-1, 3)
    return depth_points_to_reference(points, calibration, pose)


def depth_points_to_reference(points, calibration, pose):
    """Measured depth-camera points -> reference, respecting PnP pixel axes."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return points
    transform = calibration.depth_to_color
    points = points @ transform[:3, :3].T + transform[:3, 3]
    camera = calibration.color
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.K, camera.D)
    pixels = pose[1].to_image_points(pixels).astype(np.float64).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pixels, camera.K, camera.D).reshape(-1, 2)
    pnp_points = np.column_stack((xy*points[:, 2:3], points[:, 2]))
    camera_to_reference = np.linalg.inv(pose[0])
    return pnp_points @ camera_to_reference[:3, :3].T + camera_to_reference[:3, 3]


def reference_points_to_depth(points, calibration, pose):
    """Inverse of depth_points_to_reference, including the board pixel mapping."""
    if not len(points):
        return np.empty((0, 3))
    pixels, z = reference_points_to_image(points, pose, calibration.color)
    camera = calibration.color
    xy = cv2.undistortPoints(pixels.astype(np.float64).reshape(-1, 1, 2), camera.K, camera.D).reshape(-1, 2)
    points = np.column_stack((xy*z[:, None], z))
    transform = np.linalg.inv(calibration.depth_to_color)
    return points @ transform[:3, :3].T + transform[:3, 3]
