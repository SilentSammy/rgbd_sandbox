"""Window keys: 1 left + depth; 2 center RGB; 3 right + depth; U undistort; M sparse/dense; Q quit."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from stretch4_body.subsystem.cameras import RGBPipelineController, RGBCameras
from stretch4_body.subsystem.cameras.emulated_rgbd import (
    create_rgbd_frame, get_lidar_stream, transform_and_optionally_unify_clouds,
)
from stretch4_emulated_rgbd.api import DenseDepthImage, ExtrinsicsCalibration
from stretch4_emulated_rgbd.dual_lidar_calibration import DualLidarCalibration


def depth_heatmap(depth, near, far):
    valid = np.isfinite(depth) & (depth > 0)
    values = np.zeros(depth.shape, dtype=np.uint8)
    values[valid] = np.clip((depth[valid]-near)*255/(far-near), 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def display_depth(depth, max_dimension):
    """Reduce measured samples without losing them; keep the nearest sample per pixel."""
    h, w = depth.shape
    scale = min(1, max_dimension/max(h, w))
    if scale < 1:
        height, width = round(h*scale), round(w*scale)
        y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
        reduced = np.full((height, width), np.inf, np.float32)
        np.minimum.at(reduced, (y*height//h, x*width//w), depth[y, x])
        reduced[~np.isfinite(reduced)] = 0
        depth = reduced
    return depth


def rectification_maps(calibration, size):
    """Pinhole view with no 3D rotation, so camera-Z depth stays unchanged."""
    matrix = calibration.camera_matrix
    distortion = calibration.distortion_coefficients
    new_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        matrix, distortion, size, np.eye(3), balance=0)
    return cv2.fisheye.initUndistortRectifyMap(
        matrix, distortion, np.eye(3), new_matrix, size, cv2.CV_32FC1)


def main(max_dimension=800, depth_min=0.25, depth_max=2.0, dense=False, undistort=False):
    camera = RGBPipelineController(
        camera_type=RGBCameras.head_left_right_center_compressed,
        recording_directory=None, show_image_in=None,
        is_rotate=True, is_rectify=False, is_crop=False,
        ai_models_to_use=[], detect_aruco_marker_size=None,
        is_open_camera=False,
    )
    names = {'1': 'left', '2': 'center', '3': 'right'}
    types = dict(zip(('left', 'center', 'right'),
                     (RGBCameras.left(), RGBCameras.center(), RGBCameras.right())))
    calibrations = {name: kind.load_calibration() for name, kind in types.items()}
    calibration_dir = Path(os.environ['HELLO_FLEET_PATH']) / os.environ['HELLO_FLEET_ID'] / 'calibration_cameras'
    # Use the same factory baseline and base frame as the optimization tools.
    extrinsics = yaml.safe_load((calibration_dir / 'camera_extrinsics.yaml').read_text())
    lidar_calibration = DualLidarCalibration()
    lidar_transforms = [lidar_calibration.get_lidar_to_base_transform(is_right_lidar=right)
                        for right in (False, True)]
    base_to_center = np.array(extrinsics['transform_right_lidar_to_head_center']['data']) @ np.linalg.inv(lidar_transforms[1])
    camera_transforms = {'center': base_to_center}
    for name in ('left', 'right'):
        camera_transforms[name] = np.linalg.inv(extrinsics[f'{name}_to_center']) @ base_to_center
    for name in types:
        path = calibration_dir / f'emulated_rgbd_extrinsics_{name}_camera_both_lidar.yaml'
        if path.exists():
            correction = ExtrinsicsCalibration.load_from_yaml(path)
            camera_transforms[name] = correction.apply_to_camera_extrinsics(camera_transforms[name])
    lidar = get_lidar_stream()
    selected = 'left'
    maps = {}
    print(__doc__, flush=True)
    print('Focus any viewer window to enter commands.', flush=True)
    print(f'Depth mode: {"dense (interpolated)" if dense else "sparse (measured)"}', flush=True)
    try:
        camera.open_camera()
        for frames in camera.get_frame_synced(is_run_pipeline=True):
            if frames is None:
                continue
            frame = getattr(frames, selected)
            if frame is None:
                continue
            image = frame.image
            depth = None
            if selected != 'center':
                rotation = types[selected].config.rotate_number_of_times
                pair = next(lidar)
                if pair is not None:
                    points = transform_and_optionally_unify_clouds(
                        pair[0].points, pair[1].points, *lidar_transforms)
                    rgbd = create_rgbd_frame(types[selected], frame, points,
                                             camera_transforms[selected], calibrations[selected])
                    # Projection uses raw pixels; apply the RGB camera's rotation to depth too.
                    depth = np.rot90(rgbd.depth_image, rotation)
                    if dense:
                        depth = DenseDepthImage(
                            image, depth, camera_name=selected,
                            lidar_name='both_lidar').compute_dense_depth()
                if undistort:
                    size = frame.image_raw.shape[1::-1]
                    map_key = (selected, size)
                    if map_key not in maps:
                        maps[map_key] = rectification_maps(calibrations[selected], size)
                    map_x, map_y = maps[map_key]
                    image = np.rot90(cv2.remap(frame.image_raw, map_x, map_y, cv2.INTER_LINEAR), rotation)
                    if depth is not None:
                        depth = np.rot90(cv2.remap(np.rot90(depth, -rotation), map_x, map_y,
                                                  cv2.INTER_NEAREST), rotation)
                if depth is not None and not dense:
                    depth = display_depth(depth, max_dimension)
            images = {'RGB': image}
            if selected != 'center':
                if depth is None:
                    depth = np.zeros(image.shape[:2], np.float32)
                images['Depth'] = depth_heatmap(depth, depth_min, depth_max)
            for window, shown in images.items():
                h, w = shown.shape[:2]
                scale = min(1, max_dimension/max(h, w))
                if scale < 1:
                    shown = cv2.resize(shown, (round(w*scale), round(h*scale)))
                cv2.imshow(window, shown)
            cv2.setWindowTitle('RGB', f'{selected} RGB | {"undistorted" if undistort and selected != "center" else "original"} | 1/2/3 camera, U undistort, M {"dense" if dense else "sparse"}')
            key = chr(cv2.waitKey(1) & 0xff).lower()
            if key in ('q', '\x1b') or cv2.getWindowProperty('RGB', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 'm':
                dense = not dense
            if key == 'u':
                undistort = not undistort
            if key in names:
                selected = names[key]
                if selected == 'center' and 'Depth' in images:
                    cv2.destroyWindow('Depth')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            lidar.close()
        finally:
            try:
                camera.stop()
            finally:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-dimension', type=int, default=800)
    parser.add_argument('--depth-min', type=float, default=0.25, help='Heatmap near distance (meters)')
    parser.add_argument('--depth-max', type=float, default=2.0, help='Heatmap far distance (meters)')
    parser.add_argument('--dense', action='store_true', help='Start with interpolated depth display')
    parser.add_argument('--undistort', action='store_true', help='Start with pinhole views of the side cameras')
    args = parser.parse_args()
    if args.max_dimension <= 0:
        parser.error('max-dimension must be positive')
    if not 0 <= args.depth_min < args.depth_max < float('inf'):
        parser.error('require 0 <= depth-min < depth-max, both finite')
    main(args.max_dimension, args.depth_min, args.depth_max, args.dense, args.undistort)
