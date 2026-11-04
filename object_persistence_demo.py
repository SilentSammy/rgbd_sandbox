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

from board_pose_setup import DEFAULT_ROTATIONS
from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _resize, _display_pixel
from rgbd_object_persistence import ObjectMemory
from rgbd_geometry import (
    reference_points_to_image as predict_points,
    object_positions_in_reference as board_positions,
)
from rgbd_perception import PerceptionPipeline, ObjectPersistor, TabletopSegmenter
from rgbd_projection import Projector
from rgbd_tabletop_segmentation import bounding_contour
from tabletop_segmentation_demo import OBJECT_COLORS


def annotate(result, calibration, projector, *, max_dimension=None):
    """Display-only projection/drawing; the perception pipeline owns processing."""
    frame, pose = result.frame, result.pose
    segmentation, tracks = result.segmentation, result.tracks
    objects = () if segmentation is None else segmentation.objects
    color = _resize(cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR), max_dimension)
    valid = np.isfinite(frame.depth) & (frame.depth > 0)
    values = np.clip((np.where(valid, frame.depth, 0.25) - 0.25) / 1.75 * 255, 0, 255)
    depth = cv2.applyColorMap(values.astype(np.uint8), cv2.COLORMAP_TURBO)
    depth[~valid] = 0
    labels = np.full(color.shape[:2], -1, np.int32)
    if segmentation is not None and pose is not None:
        projected_depth, depth_labels = frame.depth, segmentation.object_labels
        if projector.depth_shape != frame.depth.shape:
            size = projector.depth_shape[1], projector.depth_shape[0]
            projected_depth = cv2.resize(frame.depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
            depth_labels = cv2.resize(depth_labels, size, interpolation=cv2.INTER_NEAREST_EXACT)
        _, inverse = projector.build_maps(projected_depth)
        valid = inverse[:, :, 0] >= 0
        x, y = inverse[valid].T
        labels[valid] = depth_labels[y, x]
    padding = round(segmentation.bottom_padding_pixels * color.shape[0] / depth.shape[0]) if segmentation else 0
    for track in tracks:
        tint = tuple(map(int, OBJECT_COLORS[track.id % len(OBJECT_COLORS)]))
        if not track.confirmed:
            tint = (0, 255, 255)
        if track.visible:
            obj = objects[track.detection_index]
            text = f"ID {track.id}"
            if not track.confirmed:
                text += f" ? {track.seen_seconds:.1f}s"
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
            if zs[0] > 0 and 0 <= u < frame.color.shape[1] and 0 <= v < frame.color.shape[0]:
                point = _display_pixel(u, v, color.shape, frame.color.shape)
                cv2.drawMarker(color, point, tint, cv2.MARKER_CROSS, 17, 2)
                cv2.putText(color, f"ID {track.id} missing {track.missing_seconds:.1f}s",
                            (point[0]+9, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tint, 1)
    return color, depth


def view(source, *, start_frame=137, max_dimension=960, memory=None, segmentation_scale=.5):
    memory = ObjectMemory() if memory is None else memory
    playback = Playback(source, start_frame)
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max-dimension must be positive")
    c = source.calibration
    pipeline = PerceptionPipeline(c, persistor=ObjectPersistor(c, memory), allow_reverse=True,
                                  segmenter=TabletopSegmenter(c.depth, processing_scale=segmentation_scale))
    # Project for the actual display size, not the much larger native RGB grid.
    scale = 1 if max_dimension is None else max_dimension/max(c.color.height, c.color.width)
    color_shape = (max(1, round(c.color.height*scale)), max(1, round(c.color.width*scale)))
    depth_shape = (c.depth.height//2, c.depth.width//2) if segmentation_scale == .5 else (c.depth.height, c.depth.width)
    display_K, depth_K = c.color.K.copy(), c.depth.K.copy()
    for K, camera, shape in ((display_K, c.color, color_shape), (depth_K, c.depth, depth_shape)):
        sx, sy = shape[1]/camera.width, shape[0]/camera.height
        K[0] *= sx
        K[1] *= sy
        K[0, 2] += (sx-1)*.5
        K[1, 2] += (sy-1)*.5
    projector = Projector(depth_K, display_K, c.depth_to_color,
                          depth_shape, color_shape,
                          depth_D=c.depth.D, color_D=c.color.D,
                          depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)
    displayed = None
    windows = ("RGB object persistence", "Depth object persistence")
    print(PLAYBACK_HELP + " | C clear memory")
    print(f"Segmentation scale: {segmentation_scale}; masks/annotations remain native-size")
    try:
        for name in windows:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        while True:
            index = playback.index
            if index != displayed:
                frame = source[index]
                result = pipeline.process(frame, frame_index=index)
                started = perf_counter()
                color, depth = annotate(result, c, projector, max_dimension=max_dimension)
                display_ms = (perf_counter()-started)*1000
                dt, status, ms = result.dt, result.status, result.timings_ms["total"]
                visible = sum(t.visible and t.confirmed for t in result.tracks)
                pending = sum(not t.confirmed for t in result.tracks)
                missing = sum(not t.visible for t in result.tracks)
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
                print(f"Frame {index}: dt {dt:.3f}s | perception {ms:.1f} ms | display {display_ms:.1f} ms | "
                      f"visible IDs {[t.id for t in result.tracks if t.visible and t.confirmed]} | "
                      f"{pending} probation, {missing} missing | {status}", flush=True)
                displayed = index
            command = playback.poll(1 if playback.direction else 20)
            try:
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows):
                    break
            except cv2.error:
                break
            if playback.quit:
                break
            if command == "c":
                pipeline.persistor.clear()
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
    parser.add_argument("--segmentation-scale", type=float, choices=(.5, 1.), default=.5,
                        help="depth processing scale (default: 0.5); output pixel coordinates stay native")
    parser.add_argument("--probation-period", type=float, default=1.0, help="seconds of consecutive detection")
    parser.add_argument("--match-distance", type=float, default=0.12, help="maximum 3D association distance in meters")
    parser.add_argument("--forget-after", type=float, default=5.0, help="seconds without detection before forgetting")
    args = parser.parse_args()
    memory = ObjectMemory(match_distance=args.match_distance, probation_period=args.probation_period,
                          forget_after=args.forget_after)
    with RealSenseReader(args.bag, rotations=args.rotations) as source:
        view(source, start_frame=args.start_frame, max_dimension=args.max_dimension, memory=memory,
             segmentation_scale=args.segmentation_scale)
