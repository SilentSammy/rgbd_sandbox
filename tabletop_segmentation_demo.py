"""Browse a recorded RGB-D feed with classical tabletop segmentation.

Example::

    python tabletop_segmentation_demo.py recordings/20260924_113104.db3

The original ``rgbd_db3_viewer.py`` remains independent of this experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from rgbd_db3_reader import DepthColorProjector, RGBDDB3Reader, RGBDFrame
from rgbd_tabletop_segmentation import (
    TabletopSegmentationParameters,
    TabletopSegmentationResult,
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
)
TABLE_COLOR = np.asarray((180, 180, 180), dtype=np.uint8)


def _colorize_depth_labels(
    labels: np.ndarray,
    table_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = np.zeros((*labels.shape, 3), dtype=np.uint8)
    visible = table_mask.copy()
    overlay[table_mask] = TABLE_COLOR
    for label in np.unique(labels[labels >= 0]):
        mask = labels == label
        overlay[mask] = OBJECT_COLORS[int(label) % len(OBJECT_COLORS)]
        visible |= mask
    return overlay, visible


def _project_overlay_to_color(
    result: TabletopSegmentationResult,
    projector: DepthColorProjector,
    color_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    color_height, color_width = color_shape
    overlay = np.zeros((color_height, color_width, 3), dtype=np.uint8)
    visible = np.zeros((color_height, color_width), dtype=bool)

    source_labels = result.object_labels.ravel()
    source_table = result.table_mask.ravel()
    source_indices = projector.color_to_depth.ravel()
    has_source = source_indices >= 0

    projected_labels = np.full(source_indices.shape, -1, dtype=np.int32)
    projected_table = np.zeros(source_indices.shape, dtype=bool)
    projected_labels[has_source] = source_labels[source_indices[has_source]]
    projected_table[has_source] = source_table[source_indices[has_source]]
    projected_labels = projected_labels.reshape(color_shape)
    projected_table = projected_table.reshape(color_shape)

    # A depth sample covers several pixels at the higher color resolution.
    # Dilating individual masks produces a readable overlay without changing
    # the calibrated depth-to-color projection itself.
    kernel = np.ones((3, 3), dtype=np.uint8)
    table_dilated = cv2.dilate(projected_table.astype(np.uint8), kernel) != 0
    overlay[table_dilated] = TABLE_COLOR
    visible |= table_dilated
    for label in np.unique(projected_labels[projected_labels >= 0]):
        mask = cv2.dilate(
            (projected_labels == label).astype(np.uint8), kernel
        ) != 0
        overlay[mask] = OBJECT_COLORS[int(label) % len(OBJECT_COLORS)]
        visible |= mask
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


def _print_result(frame: RGBDFrame, result: TabletopSegmentationResult) -> None:
    a, b, c, d = result.plane_model
    timings = ", ".join(
        f"{name}={duration:.1f} ms" for name, duration in result.timings_ms.items()
    )
    print(
        f"frame {frame.index}: {len(result.objects)} objects | "
        f"valid={result.valid_pixel_count:,}, "
        f"voxels={result.downsampled_point_count:,}, "
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


def _draw_result(
    frame: RGBDFrame,
    result: TabletopSegmentationResult,
    axes: np.ndarray,
    projector: DepthColorProjector,
) -> float:
    if frame.color is None or frame.depth_meters is None:
        raise ValueError("the demo requires paired color and depth frames")

    projection_started = perf_counter()
    projector.build(frame.depth_meters, frame.color.shape[:2])
    color_overlay, color_mask = _project_overlay_to_color(
        result, projector, frame.color.shape[:2]
    )
    projection_ms = (perf_counter() - projection_started) * 1000

    depth_overlay, depth_mask = _colorize_depth_labels(
        result.object_labels, result.table_mask
    )
    segmented_depth = np.zeros_like(depth_overlay)
    segmented_depth[depth_mask] = depth_overlay[depth_mask]

    for axis in axes:
        axis.clear()
        axis.axis("off")

    axes[0].imshow(_blend(frame.color, color_overlay, color_mask))
    axes[0].set_title("Calibrated color overlay")

    axes[1].imshow(
        np.ma.masked_less_equal(frame.depth_meters, 0),
        cmap="turbo",
        vmin=0.25,
        vmax=2.0,
    )
    axes[1].imshow(depth_overlay, alpha=np.where(depth_mask, 0.50, 0.0))
    axes[1].set_title("Depth + table/object masks")

    axes[2].imshow(segmented_depth)
    axes[2].set_title(f"{len(result.objects)} object clusters")
    for obj in result.objects:
        u, v = obj.centroid_depth_pixel
        axes[2].text(
            u,
            v,
            str(obj.label),
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.7, "edgecolor": "none"},
        )

    axes[0].figure.suptitle(
        f"Frame {frame.index} | segmentation {result.timings_ms['total']:.1f} ms "
        f"| projection/overlay {projection_ms:.1f} ms"
    )
    axes[0].figure.canvas.draw_idle()
    plt.pause(0.001)
    return projection_ms


def run_demo(
    bag_path: Path,
    parameters: TabletopSegmentationParameters,
    *,
    initial_frame: int = 0,
    no_gui: bool = False,
) -> None:
    with RGBDDB3Reader(bag_path, rotate_clockwise=True) as reader:
        if len(reader) == 0:
            raise ValueError("the recording contains no paired RGB-D frames")
        if not 0 <= initial_frame < len(reader):
            raise ValueError(f"frame must be between 0 and {len(reader) - 1}")
        if reader.depth_intrinsics is None:
            raise ValueError("the recording does not contain depth intrinsics")

        projector = reader.make_projector()
        index = initial_frame
        axes: np.ndarray | None = None
        if not no_gui:
            _, axes = plt.subplots(1, 3, figsize=(16, 7))
            plt.show(block=False)
            print("Commands: Enter/n = next, p = previous, <number> = jump, q = quit")

        while True:
            reader.seek(index)
            ok, frame = reader.read()
            if not ok or frame is None or frame.depth_meters is None:
                break

            result = segment_tabletop_objects(
                frame.depth_meters,
                reader.depth_intrinsics,
                parameters,
            )
            _print_result(frame, result)

            if no_gui:
                break
            assert axes is not None
            _draw_result(frame, result, axes, projector)
            if not plt.fignum_exists(axes[0].figure.number):
                break

            command = input(f"frame {index}/{len(reader) - 1}> ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            if command in {"", "n", "next"}:
                index = min(index + 1, len(reader) - 1)
            elif command in {"p", "prev", "previous"}:
                index = max(index - 1, 0)
            else:
                try:
                    requested = int(command)
                except ValueError:
                    print("Unknown command. Use Enter/n, p, a frame number, or q.")
                    continue
                if 0 <= requested < len(reader):
                    index = requested
                else:
                    print(f"Frame must be between 0 and {len(reader) - 1}.")

    plt.close("all")


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment tabletop objects in a recorded RGB-D stream."
    )
    parser.add_argument(
        "bag",
        nargs="?",
        type=Path,
        default=Path("recordings/20260924_113104.db3"),
        help="recording path (default: recordings/20260924_113104.db3)",
    )
    parser.add_argument("--frame", type=int, default=0, help="initial frame index")
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
        object_min_height=args.object_min_height,
        object_max_height=args.object_max_height,
        min_object_peak_height=args.min_object_peak_height,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        min_object_pixels=args.min_object_pixels,
        border_margin_pixels=args.border_margin_pixels,
        split_merged_clusters=not args.no_split_merged,
        merged_cluster_min_width=args.merged_cluster_min_width,
        split_valley_ratio=args.split_valley_ratio,
    )
    run_demo(
        args.bag,
        parameters,
        initial_frame=args.frame,
        no_gui=args.no_gui,
    )


if __name__ == "__main__":
    main()
