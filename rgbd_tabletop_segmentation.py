"""Inspectable, geometry-only tabletop object segmentation for RGB-D frames.

The implementation deliberately keeps the calibration and image coordinate
system supplied by :mod:`rgbd_db3_reader`.  It uses Open3D for voxel
downsampling, RANSAC plane fitting, and DBSCAN clustering, while retaining the
source depth-pixel indices needed to turn 3-D clusters back into image masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import open3d as o3d

from rgbd_db3_reader import CameraIntrinsics


@dataclass(frozen=True)
class TabletopSegmentationParameters:
    """Metric thresholds controlling the transparent segmentation pipeline."""

    min_depth: float = 0.25
    max_depth: float = 2.0
    voxel_size: float = 0.005
    plane_roi_top: float = 0.40
    object_roi_top: float = 0.50
    object_roi_left: float = 0.05
    object_roi_right: float = 0.90
    plane_distance_threshold: float = 0.010
    ransac_iterations: int = 1_000
    plane_candidate_count: int = 4
    plane_min_inliers: int = 1_000
    plane_min_inlier_fraction: float = 0.25
    support_normal_y_min: float = 0.75
    object_min_height: float = 0.010
    object_max_height: float = 0.40
    min_object_peak_height: float = 0.050
    cluster_eps: float = 0.015
    cluster_min_points: int = 10
    min_object_pixels: int = 500
    border_margin_pixels: int = 2
    split_merged_clusters: bool = True
    merged_cluster_min_width: float = 0.10
    split_valley_ratio: float = 0.20
    split_min_child_pixels: int = 500
    split_min_child_width_pixels: int = 10
    random_seed: int = 0

    def validate(self) -> None:
        """Raise ``ValueError`` for parameter combinations that cannot work."""

        if not 0 < self.min_depth < self.max_depth:
            raise ValueError("depth limits must satisfy 0 < min_depth < max_depth")
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        if not 0 <= self.plane_roi_top < 1:
            raise ValueError("plane_roi_top must be in the range [0, 1)")
        if not 0 <= self.object_roi_top < 1:
            raise ValueError("object_roi_top must be in the range [0, 1)")
        if not 0 <= self.object_roi_left < self.object_roi_right <= 1:
            raise ValueError(
                "object ROI must satisfy 0 <= left < right <= 1"
            )
        if self.plane_distance_threshold <= 0:
            raise ValueError("plane_distance_threshold must be positive")
        if self.ransac_iterations <= 0:
            raise ValueError("ransac_iterations must be positive")
        if self.plane_candidate_count <= 0 or self.plane_min_inliers < 3:
            raise ValueError("plane candidate count and minimum inliers must be positive")
        if not 0 < self.plane_min_inlier_fraction <= 1:
            raise ValueError("plane_min_inlier_fraction must be in the range (0, 1]")
        if not 0 <= self.support_normal_y_min <= 1:
            raise ValueError("support_normal_y_min must be in the range [0, 1]")
        if not 0 <= self.object_min_height < self.object_max_height:
            raise ValueError(
                "object heights must satisfy 0 <= min_height < max_height"
            )
        if not self.object_min_height <= self.min_object_peak_height <= self.object_max_height:
            raise ValueError(
                "min_object_peak_height must lie inside the object height range"
            )
        if self.cluster_eps <= 0:
            raise ValueError("cluster_eps must be positive")
        if self.cluster_min_points <= 0 or self.min_object_pixels <= 0:
            raise ValueError("cluster point thresholds must be positive")
        if self.border_margin_pixels < -1:
            raise ValueError("border_margin_pixels must be -1 or greater")
        if self.merged_cluster_min_width <= 0:
            raise ValueError("merged_cluster_min_width must be positive")
        if not 0 <= self.split_valley_ratio <= 1:
            raise ValueError("split_valley_ratio must be in the range [0, 1]")
        if self.split_min_child_pixels <= 0 or self.split_min_child_width_pixels <= 0:
            raise ValueError("split child thresholds must be positive")


@dataclass(frozen=True)
class DetectedObject:
    """Geometric summary for one accepted point cluster."""

    label: int
    pixel_count: int
    voxel_count: int
    centroid_xyz: tuple[float, float, float]
    min_bound_xyz: tuple[float, float, float]
    max_bound_xyz: tuple[float, float, float]
    extent_xyz: tuple[float, float, float]
    max_height_above_plane: float
    centroid_depth_pixel: tuple[float, float]


@dataclass(frozen=True)
class TabletopSegmentationResult:
    """Plane, pixel masks, object geometry, and performance diagnostics."""

    plane_model: tuple[float, float, float, float]
    table_mask: np.ndarray
    object_labels: np.ndarray
    objects: tuple[DetectedObject, ...]
    valid_pixel_count: int
    downsampled_point_count: int
    table_voxel_count: int
    candidate_voxel_count: int
    noise_voxel_count: int
    timings_ms: dict[str, float]


def _scaled_intrinsics(
    intrinsics: CameraIntrinsics,
    shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    height, width = shape
    calibrated_aspect = intrinsics.width / intrinsics.height
    actual_aspect = width / height
    if not np.isclose(actual_aspect, calibrated_aspect, rtol=1e-3):
        raise ValueError(
            f"depth shape {shape} is incompatible with calibrated size "
            f"({intrinsics.height}, {intrinsics.width})"
        )

    scale_x = width / intrinsics.width
    scale_y = height / intrinsics.height
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    cx = (intrinsics.cx + 0.5) * scale_x - 0.5
    cy = (intrinsics.cy + 0.5) * scale_y - 0.5
    return fx, fy, cx, cy


def _as_float_tuple(values: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _split_at_horizontal_valley(
    member_indices: np.ndarray,
    pixel_x: np.ndarray,
    points: np.ndarray,
    params: TabletopSegmentationParameters,
) -> tuple[np.ndarray, ...]:
    """Split a suspiciously wide cluster at a strong vertical image gap.

    DBSCAN can join adjacent cartons through a thin chain of depth samples. A
    true two-carton merge has two dense horizontal lobes with a sparse valley;
    a single wide carton generally does not. The conservative tests below make
    this a no-op unless that structure is clearly present.
    """

    if not params.split_merged_clusters:
        return (member_indices,)
    member_points = points[member_indices]
    if float(np.ptp(member_points[:, 0])) < params.merged_cluster_min_width:
        return (member_indices,)

    member_x = pixel_x[member_indices]
    min_x = int(member_x.min())
    max_x = int(member_x.max())
    width = max_x - min_x + 1
    child_width = params.split_min_child_width_pixels
    if width < 2 * child_width + 1:
        return (member_indices,)

    column_counts = np.bincount(member_x - min_x, minlength=width).astype(np.float64)
    smoothed = np.convolve(column_counts, np.ones(3) / 3.0, mode="same")
    cumulative = np.cumsum(column_counts)
    candidate_columns = np.arange(child_width, width - child_width)
    enough_pixels = (
        (cumulative[candidate_columns] >= params.split_min_child_pixels)
        & (
            cumulative[-1] - cumulative[candidate_columns]
            >= params.split_min_child_pixels
        )
    )
    candidate_columns = candidate_columns[enough_pixels]
    if not len(candidate_columns):
        return (member_indices,)

    valley_index = int(candidate_columns[np.argmin(smoothed[candidate_columns])])
    left_peak = float(smoothed[:valley_index].max(initial=0.0))
    right_peak = float(smoothed[valley_index + 1 :].max(initial=0.0))
    reference_peak = min(left_peak, right_peak)
    if (
        reference_peak <= 0
        or float(smoothed[valley_index])
        > params.split_valley_ratio * reference_peak
    ):
        return (member_indices,)

    cut_x = min_x + valley_index
    left = member_indices[member_x <= cut_x]
    right = member_indices[member_x > cut_x]
    if (
        len(left) < params.split_min_child_pixels
        or len(right) < params.split_min_child_pixels
    ):
        return (member_indices,)
    return left, right


def segment_tabletop_objects(
    depth_meters: np.ndarray,
    intrinsics: CameraIntrinsics,
    parameters: TabletopSegmentationParameters | None = None,
) -> TabletopSegmentationResult:
    """Separate a dominant support plane and cluster objects above it.

    The table normal is oriented toward the camera origin. Consequently,
    positive signed distances point from the table toward the camera, which is
    the expected side for objects resting on a visible tabletop.
    """

    params = parameters or TabletopSegmentationParameters()
    params.validate()

    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_meters must be a 2-D array, got {depth.shape}")

    total_started = perf_counter()
    timings: dict[str, float] = {}

    stage_started = perf_counter()
    valid_mask = (
        np.isfinite(depth)
        & (depth >= params.min_depth)
        & (depth <= params.max_depth)
    )
    flat_pixel_indices = np.flatnonzero(valid_mask.ravel())
    if len(flat_pixel_indices) < 3:
        raise ValueError("fewer than three depth pixels survive the depth limits")

    height, width = depth.shape
    pixel_y, pixel_x = np.unravel_index(flat_pixel_indices, depth.shape)
    z = depth.ravel()[flat_pixel_indices]
    fx, fy, cx, cy = _scaled_intrinsics(intrinsics, depth.shape)
    points = np.column_stack(
        (
            (pixel_x.astype(np.float32) - cx) * z / fx,
            (pixel_y.astype(np.float32) - cy) * z / fy,
            z,
        )
    ).astype(np.float64, copy=False)
    timings["unproject"] = (perf_counter() - stage_started) * 1000

    stage_started = perf_counter()
    full_cloud = o3d.geometry.PointCloud()
    full_cloud.points = o3d.utility.Vector3dVector(points)
    margin = params.voxel_size * 0.5
    min_bound = points.min(axis=0) - margin
    max_bound = points.max(axis=0) + margin
    down_cloud, _, trace = full_cloud.voxel_down_sample_and_trace(
        params.voxel_size,
        min_bound,
        max_bound,
        False,
    )
    down_points = np.asarray(down_cloud.points)
    if len(down_points) < 3:
        raise ValueError("fewer than three points remain after voxel downsampling")
    timings["downsample"] = (perf_counter() - stage_started) * 1000

    stage_started = perf_counter()
    # The recording includes a large back wall. Restricting plane hypotheses to
    # the lower part of the image makes "dominant plane" mean the visible
    # support surface, not the largest plane anywhere in the room. The filter
    # is expressed through the calibrated projection so it remains valid after
    # voxel downsampling.
    with np.errstate(divide="ignore", invalid="ignore"):
        down_pixel_x = fx * down_points[:, 0] / down_points[:, 2] + cx
        down_pixel_y = fy * down_points[:, 1] / down_points[:, 2] + cy
    plane_fit_indices = np.flatnonzero(
        np.isfinite(down_pixel_y) & (down_pixel_y >= params.plane_roi_top * height)
    )
    if len(plane_fit_indices) < 3:
        raise ValueError("fewer than three points lie inside the plane-fitting ROI")
    plane_fit_cloud = down_cloud.select_by_index(plane_fit_indices.tolist())

    # A scene can contain several parallel horizontal surfaces. In particular,
    # some recordings show both the medicine table and a large structure at
    # the bottom of the image. Peel off several RANSAC planes, keep substantial
    # planes whose normal points mostly along the rotated image's vertical
    # axis, and choose the nearest credible one rather than blindly choosing
    # the globally largest plane.
    candidates: list[tuple[np.ndarray, int]] = []
    remaining_plane_cloud = plane_fit_cloud
    for candidate_index in range(params.plane_candidate_count):
        if len(remaining_plane_cloud.points) < 3:
            break
        o3d.utility.random.seed(params.random_seed + candidate_index)
        raw_plane, inliers = remaining_plane_cloud.segment_plane(
            distance_threshold=params.plane_distance_threshold,
            ransac_n=3,
            num_iterations=params.ransac_iterations,
        )
        if len(inliers) < 3:
            break
        candidate = np.asarray(raw_plane, dtype=np.float64)
        normal_length = float(np.linalg.norm(candidate[:3]))
        if np.isfinite(normal_length) and normal_length > 0:
            candidate /= normal_length
            # The camera is at the origin. Positive d orients the normal from
            # the support surface toward the camera.
            if candidate[3] < 0:
                candidate = -candidate
            if (
                len(inliers) >= params.plane_min_inliers
                and abs(float(candidate[1])) >= params.support_normal_y_min
            ):
                candidates.append((candidate, len(inliers)))
        remaining_plane_cloud = remaining_plane_cloud.select_by_index(
            inliers, invert=True
        )

    if not candidates:
        raise RuntimeError("RANSAC found no credible support-plane candidate")
    largest_candidate = max(inlier_count for _, inlier_count in candidates)
    substantial_candidates = [
        item
        for item in candidates
        if item[1] >= largest_candidate * params.plane_min_inlier_fraction
    ]
    plane, table_voxel_count = min(
        substantial_candidates,
        key=lambda item: float(item[0][3]),
    )
    timings["plane_ransac"] = (perf_counter() - stage_started) * 1000

    stage_started = perf_counter()
    signed_down_distances = down_points @ plane[:3] + plane[3]
    candidate_down_indices = np.flatnonzero(
        (down_pixel_x >= params.object_roi_left * width)
        & (down_pixel_x <= params.object_roi_right * width)
        & (down_pixel_y >= params.object_roi_top * height)
        & (signed_down_distances >= params.object_min_height)
        & (signed_down_distances <= params.object_max_height)
    )
    candidate_cloud = down_cloud.select_by_index(candidate_down_indices.tolist())
    timings["height_filter"] = (perf_counter() - stage_started) * 1000

    stage_started = perf_counter()
    if len(candidate_down_indices) == 0:
        raw_candidate_labels = np.empty(0, dtype=np.int32)
    else:
        raw_candidate_labels = np.asarray(
            candidate_cloud.cluster_dbscan(
                eps=params.cluster_eps,
                min_points=params.cluster_min_points,
                print_progress=False,
            ),
            dtype=np.int32,
        )
    timings["dbscan"] = (perf_counter() - stage_started) * 1000

    stage_started = perf_counter()
    # Map each clustered voxel back through Open3D's trace to the original
    # point list. Values remain -1 for rejected points and DBSCAN noise.
    point_raw_labels = np.full(len(points), -1, dtype=np.int32)
    for candidate_offset, raw_label in enumerate(raw_candidate_labels):
        if raw_label < 0:
            continue
        down_index = int(candidate_down_indices[candidate_offset])
        source_indices = np.asarray(trace[down_index], dtype=np.int64)
        point_raw_labels[source_indices] = raw_label

    signed_full_distances = points @ plane[:3] + plane[3]
    accepted: list[tuple[np.ndarray, float, int]] = []
    for raw_label in np.unique(point_raw_labels[point_raw_labels >= 0]):
        member_indices = np.flatnonzero(point_raw_labels == raw_label)
        if len(member_indices) < params.min_object_pixels:
            continue
        if float(signed_full_distances[member_indices].max()) < params.min_object_peak_height:
            continue
        if params.border_margin_pixels >= 0:
            margin = params.border_margin_pixels
            member_x = pixel_x[member_indices]
            member_y = pixel_y[member_indices]
            if (
                int(member_x.min()) <= margin
                or int(member_x.max()) >= width - 1 - margin
                or int(member_y.min()) <= margin
                or int(member_y.max()) >= height - 1 - margin
            ):
                continue
        for part_indices in _split_at_horizontal_valley(
            member_indices, pixel_x, points, params
        ):
            voxel_keys = np.floor(points[part_indices] / params.voxel_size).astype(
                np.int64
            )
            part_voxel_count = len(np.unique(voxel_keys, axis=0))
            mean_pixel_x = float(pixel_x[part_indices].mean())
            accepted.append((part_indices, mean_pixel_x, part_voxel_count))
    accepted.sort(key=lambda item: item[1])

    point_labels = np.full(len(points), -1, dtype=np.int32)
    objects: list[DetectedObject] = []
    for label, (member_indices, _, part_voxel_count) in enumerate(accepted):
        point_labels[member_indices] = label
        member_points = points[member_indices]
        min_xyz = member_points.min(axis=0)
        max_xyz = member_points.max(axis=0)
        objects.append(
            DetectedObject(
                label=label,
                pixel_count=len(member_indices),
                voxel_count=part_voxel_count,
                centroid_xyz=_as_float_tuple(member_points.mean(axis=0)),
                min_bound_xyz=_as_float_tuple(min_xyz),
                max_bound_xyz=_as_float_tuple(max_xyz),
                extent_xyz=_as_float_tuple(max_xyz - min_xyz),
                max_height_above_plane=float(
                    signed_full_distances[member_indices].max()
                ),
                centroid_depth_pixel=(
                    float(pixel_x[member_indices].mean()),
                    float(pixel_y[member_indices].mean()),
                ),
            )
        )

    point_table_mask = np.abs(signed_full_distances) <= params.plane_distance_threshold
    table_mask = np.zeros(depth.size, dtype=bool)
    table_mask[flat_pixel_indices] = point_table_mask
    table_mask = table_mask.reshape(depth.shape)

    object_labels = np.full(depth.size, -1, dtype=np.int32)
    object_labels[flat_pixel_indices] = point_labels
    object_labels = object_labels.reshape(depth.shape)
    timings["label_mapping"] = (perf_counter() - stage_started) * 1000
    timings["total"] = (perf_counter() - total_started) * 1000

    return TabletopSegmentationResult(
        plane_model=tuple(float(value) for value in plane),  # type: ignore[arg-type]
        table_mask=table_mask,
        object_labels=object_labels,
        objects=tuple(objects),
        valid_pixel_count=len(points),
        downsampled_point_count=len(down_points),
        table_voxel_count=table_voxel_count,
        candidate_voxel_count=len(candidate_down_indices),
        noise_voxel_count=int(np.count_nonzero(raw_candidate_labels < 0)),
        timings_ms=timings,
    )
