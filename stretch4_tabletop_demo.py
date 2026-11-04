"""Capture one rectified right-camera RGB-D frame, segment it, and keep the results open."""

import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from rgbd_types import Camera, rotate_camera_clockwise
from rgbd_tabletop_segmentation import (
    TabletopSegmentationParameters, segment_tabletop_objects,
)


def rectify_snapshot(image, sparse, camera, dense):
    """Rectify RGB and camera-Z depth identically, with no change of 3D axes."""
    size = (camera.width, camera.height)
    matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera.K, camera.D, size, np.eye(3), balance=0)
    maps = cv2.fisheye.initUndistortRectifyMap(
        camera.K, camera.D, np.eye(3), matrix, size, cv2.CV_32FC1)
    return (
        cv2.remap(image, *maps, cv2.INTER_LINEAR),
        cv2.remap(sparse, *maps, cv2.INTER_NEAREST),
        cv2.remap(dense, *maps, cv2.INTER_NEAREST),
        Camera(camera.width, camera.height, matrix, np.zeros(4), 'none'),
    )


def capture_snapshot():
    from stretch4_emulated_rgbd.api import (
        DenseDepthImage, get_emulated_rgbd_stream,
    )

    streamer, frames = get_emulated_rgbd_stream(
        use_left=False, use_right=True, use_left_lidar=True, use_right_lidar=True,
        merge_lidars=True, resolution_height=800, emulated_rgbd_fps=10,
    )
    try:
        ready = 0
        for frame in frames:
            if frame is None or not np.any(frame.depth_image > 0):
                continue
            ready += 1
            if ready < 5:  # Allow startup/exposure to settle; process only the final frame.
                continue
            image, sparse = frame.image.copy(), frame.depth_image.copy()
            raw_matrix = streamer.camera_matrices['right']
            distortion = frame.distortion_coefficients
            # The API supplies upright pixels; rotate the native calibration exactly too.
            h, w = image.shape[:2]
            camera = rotate_camera_clockwise(
                Camera(h, w, np.array(raw_matrix, dtype=np.float64),
                       np.array(distortion, dtype=np.float64), 'fisheye'))
            dense = DenseDepthImage(image, sparse, camera_name='right',
                                    lidar_name='both_lidar').compute_dense_depth()
            return rectify_snapshot(image, sparse, camera, dense)
        raise RuntimeError('The RGB-D stream ended without a usable right-camera frame')
    finally:
        try:
            frames.close()
        finally:
            streamer.stop()


def depth_heatmap(depth, near, far):
    valid = np.isfinite(depth) & (depth > 0)
    values = np.zeros(depth.shape, np.uint8)
    values[valid] = np.clip((depth[valid]-near)*255/(far-near), 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def crop_snapshot(image, sparse, depth, camera, displayed_y, max_dimension):
    """Convert a full-frame display crop to native pixels and update intrinsics."""
    scale = min(1, max_dimension/max(image.shape[:2]))
    top, bottom = (round(y/scale) for y in displayed_y)
    if not 0 <= top < bottom <= camera.height:
        raise ValueError('crop-y must lie inside the full displayed image')
    matrix = camera.K.copy()
    matrix[1, 2] -= top
    cropped = Camera(camera.width, bottom-top, matrix, camera.D, camera.distortion_model)
    print(f'Display crop y={displayed_y} -> native rows {top}:{bottom}', flush=True)
    return image[top:bottom], sparse[top:bottom], depth[top:bottom], cropped


def render(image, depth, sparse, result, near, far):
    overlay = image.copy()
    overlay[result.table_mask] = (0.65*overlay[result.table_mask] + 0.35*180).astype(np.uint8)
    clusters = np.zeros_like(image)
    clusters[result.table_mask] = 100
    for obj in result.objects:
        hue = np.uint8([[[obj.label*37 % 180, 220, 255]]])
        color = tuple(int(c) for c in cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0, 0])
        mask = result.object_labels == obj.label
        overlay[mask] = (0.5*overlay[mask] + 0.5*np.array(color)).astype(np.uint8)
        clusters[mask] = color
        contour = obj.contour_depth_pixels.astype(np.int32)
        cv2.polylines(overlay, [contour], True, color, 2)
        x, y = contour[0]
        cv2.putText(overlay, f'{obj.label}: {obj.max_height_above_plane:.2f} m',
                    (int(x), max(18, int(y)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return {'RGB segmentation': overlay, 'Object clusters': clusters,
            'Depth (interpolated)': depth_heatmap(depth, near, far),
            'Depth (measured)': depth_heatmap(sparse, near, far)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, help='Reprocess a saved rectified_rgbd.npz instead of capturing')
    parser.add_argument('--output-dir', type=Path, default=Path('captures') / datetime.now().strftime('tabletop_%Y%m%d_%H%M%S'))
    parser.add_argument('--max-dimension', type=int, default=800)
    parser.add_argument('--crop-y', type=float, nargs=2, metavar=('TOP', 'BOTTOM'),
                        help='Vertical crop in full-frame displayed coordinates (using max-dimension)')
    parser.add_argument('--max-depth', type=float, default=2.5)
    parser.add_argument('--max-object-size', type=float, default=0.35)
    parser.add_argument('--measured-only', action='store_true', help='Segment only measured LiDAR depths, not interpolation')
    parser.add_argument('--no-gui', action='store_true', help='Save results and exit')
    args = parser.parse_args()
    if args.max_dimension <= 0 or args.max_depth <= 0.25 or args.max_object_size <= 0.02:
        parser.error('require max-dimension > 0, max-depth > 0.25, max-object-size > 0.02')

    if args.input:
        with np.load(args.input) as data:
            image, sparse, depth = data['rgb'], data['measured_depth'], data['depth']
            h, w = depth.shape
            camera = Camera(w, h, data['K'], np.zeros(4), 'none')
    else:
        print('Capturing one right-camera frame. Keep the robot and scene still.', flush=True)
        image, sparse, depth, camera = capture_snapshot()
        print('Capture complete; cameras and LiDAR released.', flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / 'rectified_rgbd.npz', rgb=image,
                        measured_depth=sparse, depth=depth, K=camera.K)
    cv2.imwrite(str(args.output_dir / 'rgb.png'), image)
    if args.crop_y:
        image, sparse, depth, camera = crop_snapshot(
            image, sparse, depth, camera, args.crop_y, args.max_dimension)
    params = TabletopSegmentationParameters(
        max_depth=args.max_depth, max_object_extent=args.max_object_size,
        # A cropped pinhole view need not put the table/objects in the lower half.
        plane_roi_top=0.0, object_roi_top=0.0, object_roi_left=0.0, object_roi_right=1.0,
        support_normal_y_min=0.45,  # Stretch 4's fixed head views the table at a steeper angle.
        plane_candidate_count=6, plane_min_inlier_fraction=0.15,
        min_object_pixels=60 if args.measured_only else 500,
        plane_min_inliers=100 if args.measured_only else 500,
    )
    started = perf_counter()
    result = segment_tabletop_objects(sparse if args.measured_only else depth, camera, params)
    print(f'{len(result.objects)} objects | segmentation {(perf_counter()-started)*1000:.1f} ms')
    print(f'Plane: {result.plane_model} | stage timings (ms): {result.timings_ms}')
    for obj in result.objects:
        print(f'  {obj.label}: center {obj.centroid_xyz} m | size {obj.extent_xyz} m | height {obj.max_height_above_plane:.3f} m')
    images = render(image, depth, sparse, result, params.min_depth, params.max_depth)
    for name, shown in images.items():
        cv2.imwrite(str(args.output_dir / (name.replace(' ', '_') + '.png')), shown)
    print(f'Snapshot and results saved to {args.output_dir.resolve()}', flush=True)
    if not args.no_gui:
        try:
            for name, shown in images.items():
                h, w = shown.shape[:2]
                scale = min(1, args.max_dimension/max(h, w))
                cv2.imshow(name, cv2.resize(shown, (round(w*scale), round(h*scale))))
            print('Frozen results. Q/Esc quits; no tracking or further segmentation.', flush=True)
            while cv2.waitKey(30) & 0xff not in (ord('q'), ord('Q'), 27):
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in images):
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
