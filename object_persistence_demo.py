"""Track segmented objects in the GridBoard frame, without OCR or history caches.

python object_persistence_demo.py --start-frame 137
Shared viewer playback; C clears memory. Yellow IDs are on probation. Colored
IDs are confirmed; crosses mark remembered positions of missing objects.
Reverse/restart process fresh observations with abs(timestamp - previous) dt.
Objects are assumed stationary relative to the board. No pose means no matches.
"""

import argparse
from time import perf_counter

import cv2
import numpy as np

from board_pose_setup import DEFAULT_ROTATIONS, make_estimator
from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _resize
from rgbd_object_persistence import ObjectMemory
from rgbd_point_memory_demo import remember_point, predict_points
from rgbd_projection import Projector
from rgbd_tabletop_segmentation import bounding_contour, segment_tabletop_objects
from tabletop_segmentation_demo import OBJECT_COLORS


def board_positions(objects, calibration, pose):
    """Depth cluster centers -> board, using the existing pose/pixel convention."""
    points = np.array([obj.centroid_xyz for obj in objects]).reshape(-1, 3)
    if not len(points):
        return points
    transform = calibration.depth_to_color
    points = points @ transform[:3, :3].T + transform[:3, 3]
    camera = calibration.color
    pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.K, camera.D)
    return np.array([remember_point(pixel, point[2], pose, camera)
                     for pixel, point in zip(pixels.reshape(-1, 2), points)])


def process_frame(frame, calibration, estimator, projector, memory, dt, *, frame_index=None):
    """Segment/localize/update once; return native RGB/depth annotations and status."""
    started = perf_counter()
    color = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
    pose = estimator.get_board_transform(color, frame_index=frame_index)
    error = ""
    try:
        result = segment_tabletop_objects(frame.depth, calibration.depth)
        objects = result.objects
    except RuntimeError as exc:
        result, objects, error = None, (), str(exc)
    positions = board_positions(objects, calibration, pose) if pose is not None else []
    tracks = memory.update(positions, dt)
    valid = np.isfinite(frame.depth) & (frame.depth > 0)
    values = np.clip((np.where(valid, frame.depth, 0.25) - 0.25) / 1.75 * 255, 0, 255)
    depth = cv2.applyColorMap(values.astype(np.uint8), cv2.COLORMAP_TURBO)
    depth[~valid] = 0
    labels = np.full(frame.color.shape[:2], -1, np.int32)
    if result is not None and pose is not None:
        _, inverse = projector.build_maps(frame.depth)
        valid = inverse[:, :, 0] >= 0
        x, y = inverse[valid].T
        labels[valid] = result.object_labels[y, x]
    padding = round(result.bottom_padding_pixels * color.shape[0] / depth.shape[0]) if result else 0
    for track in tracks:
        tint = tuple(map(int, OBJECT_COLORS[track.id % len(OBJECT_COLORS)]))
        if not track.confirmed:
            tint = (0, 255, 255)
        if track.visible:
            obj = objects[track.detection_index]
            text = f"ID {track.id}"
            if not track.confirmed:
                text += f" ? {track.seen_seconds:.1f}/{memory.probation_period:.1f}s"
            pixels = cv2.findNonZero((labels == obj.label).astype(np.uint8))
            rgb_contour = None if pixels is None else bounding_contour(pixels, labels.shape, padding)
            for image, contour in ((color, rgb_contour), (depth, obj.contour_depth_pixels)):
                if contour is not None:
                    cv2.polylines(image, [contour], True, tint, 2, cv2.LINE_AA)
                    x, y = map(int, contour[0])
                    cv2.putText(image, text, (x, max(20, y-7)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, tint, 1, cv2.LINE_AA)
        elif pose is not None:
            pixels, zs = predict_points([track.position], pose, calibration.color)
            u, v = pixels[0]
            if zs[0] > 0 and 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
                point = (round(float(u)), round(float(v)))
                cv2.drawMarker(color, point, tint, cv2.MARKER_CROSS, 17, 2)
                cv2.putText(color, f"ID {track.id} missing {track.missing_seconds:.1f}s",
                            (point[0]+9, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tint, 1)
    status = error or (estimator.status if pose is not None else "NO BOARD POSE - memory only")
    return color, depth, status, (perf_counter() - started) * 1000


def view(source, *, start_frame=137, max_dimension=960, memory=None):
    memory = ObjectMemory() if memory is None else memory
    playback = Playback(source, start_frame)
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max-dimension must be positive")
    c = source.calibration
    estimator = make_estimator(c.color)
    projector = Projector(c.depth.K, c.color.K, c.depth_to_color,
                          (c.depth.height, c.depth.width), (c.color.height, c.color.width),
                          depth_D=c.depth.D, color_D=c.color.D,
                          depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)
    displayed = previous_timestamp = None
    windows = ("RGB object persistence", "Depth object persistence")
    print(PLAYBACK_HELP + " | C clear memory")
    try:
        for name in windows:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        while True:
            index = playback.index
            if index != displayed:
                frame = source[index]
                dt = 0 if previous_timestamp is None else abs(frame.timestamp_ns - previous_timestamp)/1e9
                color, depth, status, ms = process_frame(frame, c, estimator, projector, memory, dt, frame_index=index)
                visible = sum(t.visible and t.confirmed for t in memory.tracks)
                pending = sum(not t.confirmed for t in memory.tracks)
                missing = sum(not t.visible for t in memory.tracks)
                lines = (f"Frame {index} | dt {dt:.3f}s | tracker time {memory.elapsed_seconds:.2f}s",
                         f"{visible} confirmed visible | {pending} probation | {missing} missing",
                         status)
                for name, image in zip(windows, (color, depth)):
                    image = _resize(image, max_dimension)
                    for row, text in enumerate(lines):
                        for tint, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                            cv2.putText(image, text, (8, 22+row*20), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4, tint, thickness, cv2.LINE_AA)
                    cv2.imshow(name, image)
                print(f"Frame {index}: dt {dt:.3f}s | {ms:.1f} ms | "
                      f"visible IDs {[t.id for t in memory.tracks if t.visible and t.confirmed]} | "
                      f"{pending} probation, {missing} missing | {status}", flush=True)
                displayed, previous_timestamp = index, frame.timestamp_ns
            command = playback.poll()
            try:
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows):
                    break
            except cv2.error:
                break
            if playback.quit:
                break
            if command == "c":
                memory.clear()
                displayed = None
    finally:
        for name in windows:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=PLAYBACK_HELP)
    parser.add_argument("bag", nargs="?", default="recordings/20261005_195417.db3")
    parser.add_argument("--start-frame", type=int, default=137)
    parser.add_argument("--max-dimension", type=int, default=960)
    parser.add_argument("--rotations", type=int, choices=range(4), default=DEFAULT_ROTATIONS)
    parser.add_argument("--probation-period", type=float, default=1.0, help="seconds of consecutive detection")
    parser.add_argument("--match-distance", type=float, default=0.12, help="maximum 3D association distance in meters")
    parser.add_argument("--forget-after", type=float, default=5.0, help="seconds without detection before forgetting")
    args = parser.parse_args()
    memory = ObjectMemory(match_distance=args.match_distance, probation_period=args.probation_period,
                          forget_after=args.forget_after)
    with RealSenseReader(args.bag, rotations=args.rotations) as source:
        view(source, start_frame=args.start_frame, max_dimension=args.max_dimension, memory=memory)
