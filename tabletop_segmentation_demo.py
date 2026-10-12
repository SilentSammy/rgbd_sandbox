"""Browse a recorded RGB-D feed with classical tabletop segmentation.

Example::

    python tabletop_segmentation_demo.py recordings/20261005_195417.db3 --frame 137

The original ``rgbd_db3_viewer.py`` remains independent of this experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from rgbd_db3_reader import Frame, RealSenseReader
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _resize
from rgbd_projection import Projector
from rgbd_tabletop_segmentation import (
    TabletopSegmentationParameters,
    TabletopSegmentationResult,
    bounding_contour,
    segment_tabletop_objects,
)


OBJECT_COLORS = np.asarray(
    [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (188, 189, 34),
        (23, 190, 207),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
    ],
    dtype=np.uint8,
)[:, ::-1].copy()  # RGB palette converted to OpenCV BGR.
TABLE_COLOR = np.asarray((180, 180, 180), dtype=np.uint8)


def _colorize_depth_labels(
    result: TabletopSegmentationResult,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = np.zeros((*result.object_labels.shape, 3), dtype=np.uint8)
    visible = result.table_mask.copy()
    overlay[result.table_mask] = TABLE_COLOR
    for obj in result.objects:
        mask = np.zeros(result.object_labels.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, obj.contour_depth_pixels, 1)
        overlay[mask != 0] = OBJECT_COLORS[obj.label % len(OBJECT_COLORS)]
        visible |= mask != 0
    return overlay, visible


def _project_overlay_to_color(
    result: TabletopSegmentationResult,
    inverse: np.ndarray,
    color_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    color_height, color_width = color_shape
    overlay = np.zeros((color_height, color_width, 3), dtype=np.uint8)
    visible = np.zeros((color_height, color_width), dtype=bool)

    has_source = inverse[:, :, 0] >= 0
    x, y = inverse[has_source].T
    projected_labels = np.full(color_shape, -1, dtype=np.int32)
    projected_table = np.zeros(color_shape, dtype=bool)
    projected_labels[has_source] = result.object_labels[y, x]
    projected_table[has_source] = result.table_mask[y, x]

    # Build the RGB rectangle from projected measured pixels, not from invented
    # depths inside the filled depth silhouette.
    kernel = np.ones((3, 3), dtype=np.uint8)
    table_dilated = cv2.dilate(projected_table.astype(np.uint8), kernel) != 0
    overlay[table_dilated] = TABLE_COLOR
    visible |= table_dilated
    padding = round(result.bottom_padding_pixels * color_height / result.object_labels.shape[0])
    for label in np.unique(projected_labels[projected_labels >= 0]):
        mask = (projected_labels == label).astype(np.uint8)
        contour = bounding_contour(cv2.findNonZero(mask), color_shape, padding)
        cv2.fillConvexPoly(mask, contour, 1)
        overlay[mask != 0] = OBJECT_COLORS[int(label) % len(OBJECT_COLORS)]
        visible |= mask != 0
    return overlay, visible


def _blend(image: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = image
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    if base.shape[2] == 4:
        base = base[:, :, :3]
    base = base.astype(np.uint8, copy=True)
    base[mask] = np.rint(0.55 * base[mask] + 0.45 * overlay[mask]).astype(np.uint8)
    return base


def _print_result(index: int, result: TabletopSegmentationResult) -> None:
    a, b, c, d = result.plane_model
    timings = ", ".join(
        f"{name}={duration:.1f} ms" for name, duration in result.timings_ms.items()
    )
    print(
        f"frame {index}: {len(result.objects)} objects | "
        f"valid={result.valid_pixel_count:,}, "
        f"plane-voxels={result.downsampled_point_count:,}, "
        f"table={result.table_voxel_count:,}, "
        f"candidates={result.candidate_voxel_count:,}, "
        f"noise={result.noise_voxel_count:,}"
    )
    print(f"  plane: {a:+.4f}x {b:+.4f}y {c:+.4f}z {d:+.4f} = 0")
    print(f"  timing: {timings}")
    for obj in result.objects:
        x, y, z = obj.centroid_xyz
        ex, ey, ez = obj.extent_xyz
        print(
            f"  object {obj.label}: pixels={obj.pixel_count:,}, "
            f"voxels={obj.voxel_count:,}, center=({x:.3f}, {y:.3f}, {z:.3f}) m, "
            f"AABB=({ex:.3f}, {ey:.3f}, {ez:.3f}) m, "
            f"height={obj.max_height_above_plane:.3f} m"
        )


def _render_result(
    frame: Frame,
    result: TabletopSegmentationResult,
    projector: Projector,
    index: int,
    max_dimension: int | None = None,
) -> dict[str, np.ndarray]:
    projection_started = perf_counter()
    _, inverse = projector.build_maps(frame.depth)
    color_overlay, color_mask = _project_overlay_to_color(
        result, inverse, frame.color.shape[:2]
    )
    projection_ms = (perf_counter() - projection_started) * 1000

    depth_overlay, depth_mask = _colorize_depth_labels(result)
    segmented_depth = depth_overlay.copy()
    valid = np.isfinite(frame.depth) & (frame.depth > 0)
    values = np.where(valid, frame.depth, 0.25)
    values = np.clip((values - 0.25) / (2.0 - 0.25) * 255, 0, 255)
    depth = cv2.applyColorMap(values.astype(np.uint8), cv2.COLORMAP_TURBO)
    depth[~valid] = 0

    for obj in result.objects:
        u, v = obj.centroid_depth_pixel
        cv2.putText(segmented_depth, str(obj.label), (round(u), round(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    images = {
        "RGB segmentation": _blend(cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR),
                                   color_overlay, color_mask),
        "Depth segmentation": _blend(depth, depth_overlay, depth_mask),
        "Object clusters": segmented_depth,
    }
    text = f"Frame {index} | {len(result.objects)} objects"
    for name, image in images.items():
        image = _resize(image, max_dimension)
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        images[name] = image
    print(f"  projection/overlay: {projection_ms:.1f} ms")
    return images


def run_demo(
    bag_path: Path,
    parameters: TabletopSegmentationParameters,
    *,
    initial_frame: int = 0,
    no_gui: bool = False,
    max_dimension: int | None = None,
) -> None:
    if max_dimension is not None and (not isinstance(max_dimension, int) or max_dimension <= 0):
        raise ValueError("max_dimension must be a positive integer or None")
    with RealSenseReader(bag_path, rotations=1) as reader:
        if len(reader) == 0:
            raise ValueError("the recording contains no paired RGB-D frames")
        if not 0 <= initial_frame < len(reader):
            raise ValueError(f"frame must be between 0 and {len(reader) - 1}")
        calibration = reader.calibration
        projector = Projector(
            calibration.depth.K, calibration.color.K, calibration.depth_to_color,
            (calibration.depth.height, calibration.depth.width),
            (calibration.color.height, calibration.color.width),
            depth_D=calibration.depth.D, color_D=calibration.color.D,
            depth_model=calibration.depth.distortion_model,
            color_model=calibration.color.distortion_model,
        )
        playback = Playback(reader, initial_frame)
        index = playback.index
        displayed = None
        windows = ("RGB segmentation", "Depth segmentation", "Object clusters")
        try:
            if not no_gui:
                for name in windows:
                    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
                print(PLAYBACK_HELP)
            while True:
                if index != displayed:
                    frame = reader[index]
                    result = segment_tabletop_objects(frame.depth, calibration.depth, parameters)
                    _print_result(index, result)
                    if no_gui:
                        break
                    images = _render_result(frame, result, projector, index, max_dimension)
                    offset = 20
                    for name, image in images.items():
                        cv2.imshow(name, image)
                        if displayed is None:
                            cv2.moveWindow(name, offset, 40)
                            offset += image.shape[1] + 20
                    displayed = index

                playback.poll()
                try:
                    if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows):
                        break
                except cv2.error:
                    break
                if playback.quit:
                    break
                index = playback.index
        finally:
            if not no_gui:
                for name in windows:
                    try:
                        cv2.destroyWindow(name)
                    except cv2.error:
                        pass


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment tabletop objects in a recorded RGB-D stream.", epilog=PLAYBACK_HELP
    )
    parser.add_argument(
        "bag",
        nargs="?",
        type=Path,
        default=Path("recordings/20260924_113104.db3"),
        help="recording path (default: recordings/20260924_113104.db3)",
    )
    parser.add_argument("--start-frame", "--frame", dest="start_frame", type=int, default=0,
                        help="initial frame index")
    parser.add_argument("--max-dimension", type=int,
                        help="set each image's longer axis in pixels (default: native size)")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="process one frame, print diagnostics, and exit",
    )
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--voxel-size", type=_positive_float, default=0.005)
    parser.add_argument(
        "--plane-roi-top",
        type=float,
        default=0.40,
        help="top edge of the plane-fitting region as a fraction of image height",
    )
    parser.add_argument(
        "--object-roi-top",
        type=float,
        default=0.50,
        help="top edge of the object-search region as a fraction of image height",
    )
    parser.add_argument("--object-roi-left", type=float, default=0.05)
    parser.add_argument("--object-roi-right", type=float, default=0.90)
    parser.add_argument("--plane-threshold", type=_positive_float, default=0.010)
    parser.add_argument("--ransac-iterations", type=int, default=1_000)
    parser.add_argument("--plane-stride", type=int, default=4,
                        help="sample every Nth row/column for plane fitting (default: 4)")
    parser.add_argument("--plane-max-points", type=int, default=12_000,
                        help="maximum voxel samples for RANSAC (default: 12000)")
    parser.add_argument("--object-min-height", type=float, default=0.010)
    parser.add_argument("--object-max-height", type=float, default=0.40)
    parser.add_argument(
        "--min-object-peak-height",
        type=float,
        default=0.050,
        help="reject clusters that never rise this far above the table",
    )
    parser.add_argument("--cluster-eps", type=_positive_float, default=0.015)
    parser.add_argument("--cluster-min-points", type=int, default=10)
    parser.add_argument("--min-object-pixels", type=int, default=500)
    parser.add_argument("--min-object-extent", type=float, default=0.02,
                        help="minimum longest measured AABB side in meters (default: 0.02)")
    parser.add_argument("--max-object-extent", type=float, default=0.35,
                        help="maximum longest measured AABB side in meters (default: 0.35)")
    parser.add_argument("--bottom-padding-pixels", type=int, default=5,
                        help="extend rectangles downward in native depth pixels (default: 5)")
    parser.add_argument(
        "--border-margin-pixels",
        type=int,
        default=2,
        help="reject clusters touching this image-edge margin; -1 disables it",
    )
    parser.add_argument(
        "--no-split-merged",
        action="store_true",
        help="disable conservative horizontal-valley splitting of wide clusters",
    )
    parser.add_argument("--merged-cluster-min-width", type=_positive_float, default=0.10)
    parser.add_argument("--split-valley-ratio", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parameters = TabletopSegmentationParameters(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        voxel_size=args.voxel_size,
        plane_roi_top=args.plane_roi_top,
        object_roi_top=args.object_roi_top,
        object_roi_left=args.object_roi_left,
        object_roi_right=args.object_roi_right,
        plane_distance_threshold=args.plane_threshold,
        ransac_iterations=args.ransac_iterations,
        plane_stride=args.plane_stride,
        plane_max_points=args.plane_max_points,
        object_min_height=args.object_min_height,
        object_max_height=args.object_max_height,
        min_object_peak_height=args.min_object_peak_height,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        min_object_pixels=args.min_object_pixels,
        min_object_extent=args.min_object_extent,
        max_object_extent=args.max_object_extent,
        bottom_padding_pixels=args.bottom_padding_pixels,
        border_margin_pixels=args.border_margin_pixels,
        split_merged_clusters=not args.no_split_merged,
        merged_cluster_min_width=args.merged_cluster_min_width,
        split_valley_ratio=args.split_valley_ratio,
    )
    run_demo(
        args.bag,
        parameters,
        initial_frame=args.start_frame,
        no_gui=args.no_gui,
        max_dimension=args.max_dimension,
    )


if __name__ == "__main__":
    main()
