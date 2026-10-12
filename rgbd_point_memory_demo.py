"""Remember clicked RGB-D points in the board frame and predict their RGB pixels.

Points are assumed stationary relative to BOARD in board_pose_setup.py.
Only the board is detected on later frames; object appearance is never tracked.
Click either image to add a point. Shared viewer playback; f/b jump, r return to
the last selection frame, u undo, c clear. Predictions also mark occluded points.
"""

import argparse
from time import perf_counter

import cv2
import numpy as np

from board_pose_setup import DEFAULT_ROTATIONS, make_estimator
from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _resize, _native_pixel, _display_pixel, _depth_pixel
from rgbd_projection import Projector


def remember_point(pixel, color_z, pose, camera):
    """Original RGB pixel + measured RGB-camera Z -> public board coordinates."""
    board_to_camera, pnp = pose
    # The estimator rotates corners before PnP; its pixel mapping is self-inverse.
    pixel = pnp.to_image_points([pixel]).astype(np.float64).reshape(1, 1, 2)
    xy = cv2.undistortPoints(pixel, camera.K, camera.D).reshape(2)
    camera_point = np.array([xy[0] * color_z, xy[1] * color_z, color_z, 1])
    return (np.linalg.inv(board_to_camera) @ camera_point)[:3]


def predict_points(points, pose, camera):
    """Board points -> original RGB pixels and RGB-camera Z, without new depth."""
    board_to_camera, pnp = pose
    camera_points = np.asarray(points) @ board_to_camera[:3, :3].T + board_to_camera[:3, 3]
    pixels, _ = cv2.projectPoints(camera_points, np.zeros(3), np.zeros(3), camera.K, camera.D)
    return pnp.to_image_points(pixels), camera_points[:, 2]


def view(source, *, max_dimension=None, jump=10, initial_frame=0):
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    if jump <= 0 or not 0 <= initial_frame < len(source):
        raise ValueError("Require a positive jump and a valid initial frame")
    calibration = source.calibration
    camera = calibration.color
    estimator = make_estimator(camera)
    projector = Projector(
        calibration.depth.K, camera.K, calibration.depth_to_color,
        (calibration.depth.height, calibration.depth.width), (camera.height, camera.width),
        depth_D=calibration.depth.D, color_D=camera.D,
        depth_model=calibration.depth.distortion_model, color_model=camera.distortion_model,
    )
    windows = ("RGB memory", "Depth memory")
    remembered = []  # (board XYZ, source frame); preserved across navigation.
    playback = Playback(source, initial_frame)
    index, displayed = playback.index, None
    panes = {}
    frame = pose = mapping = inverse = None
    pose_ms = map_ms = 0

    def draw():
        color, depth = panes[windows[0]][0].copy(), panes[windows[1]][0].copy()
        visible = 0
        if pose is not None and remembered:
            pixels, zs = predict_points([p for p, _ in remembered], pose, camera)
            for number, ((u, v), z) in enumerate(zip(pixels, zs), 1):
                if z <= 0 or not np.isfinite([u, v, z]).all():
                    continue
                if not (0 <= u < camera.width and 0 <= v < camera.height):
                    continue  # Out-of-view predictions must not clamp to the image edge.
                point = _display_pixel(u, v, color.shape, frame.color.shape)
                cv2.drawMarker(color, point, (0, 255, 255), cv2.MARKER_CROSS, 17, 2)
                cv2.putText(color, str(number), (point[0] + 9, point[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                visible += 1
        status = estimator.status if pose else "NO BOARD POSE - predictions hidden"
        text = f"Frame {index} | {status} | remembered {len(remembered)}, in view {visible}"
        cv2.putText(color, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(color, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow(windows[0], color)
        cv2.imshow(windows[1], depth)
        return {windows[0]: color, windows[1]: depth}

    def on_mouse(event, x, y, flags, name):
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN) or name not in panes:
            return
        image, native = panes[name]
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            return
        u, v = _native_pixel(x, y, image.shape, native.shape)
        if name == windows[1]:
            depth_pixel = (u, v)
        else:
            match = _depth_pixel(inverse, u, v)
            depth_pixel = None if match is None else tuple(match[0])
        if depth_pixel is None:
            if event == cv2.EVENT_LBUTTONDOWN:
                print("No measured depth near that RGB pixel; point not added.")
            return
        du, dv = depth_pixel
        meters = float(frame.depth[dv, du])
        rgb_pixel = mapping[dv, du]
        if not np.isfinite(rgb_pixel).all() or not np.isfinite(meters) or meters <= 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if pose is None:
                print("Board pose unavailable; point not added.")
                return
            # Store the actual depth sample, which may be near the RGB click.
            color_z = projector.rays[2][dv, du] * meters + projector.translation[2]
            point = remember_point(rgb_pixel, color_z, pose, camera)
            remembered.append((point, index))
            print(f"Point {len(remembered)} from frame {index}: board XYZ {point.round(4)} m")
        rendered = draw()
        target = windows[0] if name == windows[1] else windows[1]
        target_image, target_native = panes[target]
        overlay = rendered[target]
        pixel = rgb_pixel if target == windows[0] else (du, dv)
        point = _display_pixel(*pixel, target_image.shape, target_native.shape)
        cv2.drawMarker(overlay, point, (255, 0, 255), cv2.MARKER_CROSS, 15, 1)
        cv2.imshow(target, overlay)

    try:
        for name in windows:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(name, on_mouse, name)
        print("Click either image: remember point. Yellow RGB crosses are predictions.")
        print(PLAYBACK_HELP)
        print(f"F/B +/-{jump} | R last selection frame | U undo | C clear")
        while True:
            if index != displayed:
                frame = source[index]
                native_color = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
                started = perf_counter()
                pose = estimator.get_board_transform(native_color, frame_index=index)
                pose_ms = (perf_counter() - started) * 1000
                started = perf_counter()
                mapping, inverse = projector.build_maps(frame.depth)
                map_ms = (perf_counter() - started) * 1000
                valid = np.isfinite(frame.depth) & (frame.depth > 0)
                values = np.clip((np.where(valid, frame.depth, 0.25) - 0.25) / 1.75 * 255, 0, 255)
                depth = cv2.applyColorMap(values.astype(np.uint8), cv2.COLORMAP_TURBO)
                depth[~valid] = 0
                panes = {
                    windows[0]: (_resize(native_color, max_dimension), frame.color),
                    windows[1]: (_resize(depth, max_dimension), frame.depth),
                }
                draw()
                print(f"Frame {index}: pose {pose_ms:.1f} ms, map {map_ms:.1f} ms")
                if displayed is None:
                    cv2.moveWindow(windows[0], 20, 40)
                    cv2.moveWindow(windows[1], 40 + panes[windows[0]][0].shape[1], 40)
                displayed = index
            command = playback.poll()
            try:
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows):
                    break
            except cv2.error:
                break
            if playback.quit:
                break
            if command in ("f", "b"):
                playback.seek(index + (jump if command == "f" else -jump))
            elif command == "r" and remembered:
                playback.seek(remembered[-1][1])
            elif command in ("u", "c"):
                if command == "c":
                    remembered.clear()
                elif remembered:
                    remembered.pop()
                draw()
            index = playback.index
    finally:
        for name in windows:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=PLAYBACK_HELP)
    parser.add_argument("bag")
    parser.add_argument("--rotations", type=int, choices=range(4), default=DEFAULT_ROTATIONS)
    parser.add_argument("--max-dimension", type=int)
    parser.add_argument("--start-frame", "--frame", dest="start_frame", type=int, default=0)
    parser.add_argument("--jump", type=int, default=10)
    args = parser.parse_args()
    with RealSenseReader(args.bag, rotations=args.rotations) as source:
        view(source, max_dimension=args.max_dimension, jump=args.jump,
             initial_frame=args.start_frame)
