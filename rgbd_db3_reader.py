"""Small RGB-D reader for RealSense-style ROS 2 `.db3` recordings.

The API is intentionally close to OpenCV's ``VideoCapture``: construct a reader,
call ``read()`` until it returns ``False``, and then ``release()``. The yielded
frames contain both color and depth arrays.

This module is ROS-less at runtime. It reads the sqlite bag directly and uses the
pure-Python ``rosbags`` package to deserialize standard ROS 2 messages.

Pass ``rotate_clockwise=True`` to make portrait orientation the public coordinate
system. Returned arrays, active intrinsics, depth-to-color extrinsics, and pixel
correspondence maps will then all use that same rotated convention. The original
calibration remains available through the ``native_*`` reader attributes.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any, Iterator, Protocol, cast

import numpy as np
from rosbags.typesys import Stores, get_typestore  # type: ignore[reportMissingImports]


class StringMessage(Protocol):
    data: str


class ImageMessage(Protocol):
    height: int
    width: int
    encoding: str
    is_bigendian: int
    data: bytes | bytearray | np.ndarray


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole calibration for one recorded image stream."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]


@dataclass(frozen=True)
class RigidTransform:
    """A 3D rotation and translation, with translation measured in meters."""

    rotation: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]


def rotate_intrinsics_clockwise(intrinsics: CameraIntrinsics) -> CameraIntrinsics:
    """Return calibration for a 90-degree-clockwise image/camera-axis rotation."""

    coefficients = intrinsics.distortion_coefficients
    if "brown conrady" in intrinsics.distortion_model.lower() and len(coefficients) >= 4:
        # Brown-Conrady order is k1, k2, p1, p2, k3, ... . Under
        # x'=-y, y'=x, radial terms remain unchanged and (p1, p2) becomes
        # (p2, -p1).
        coefficients = (
            coefficients[0],
            coefficients[1],
            coefficients[3],
            -coefficients[2],
            *coefficients[4:],
        )
    return CameraIntrinsics(
        width=intrinsics.height,
        height=intrinsics.width,
        fx=intrinsics.fy,
        fy=intrinsics.fx,
        cx=intrinsics.height - 1 - intrinsics.cy,
        cy=intrinsics.cx,
        distortion_model=intrinsics.distortion_model,
        distortion_coefficients=coefficients,
    )


def rotate_transform_clockwise(transform: RigidTransform) -> RigidTransform:
    """Express a transform after rotating both camera axes clockwise."""

    axis_rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    rotation = np.asarray(transform.rotation, dtype=np.float64)
    translation = np.asarray(transform.translation, dtype=np.float64)
    rotated_rotation = axis_rotation @ rotation @ axis_rotation.T
    rotated_translation = axis_rotation @ translation
    return RigidTransform(
        rotation=cast(
            tuple[tuple[float, float, float], ...],
            tuple(tuple(float(value) for value in row) for row in rotated_rotation),
        ),
        translation=cast(
            tuple[float, float, float],
            tuple(float(value) for value in rotated_translation),
        ),
    )


def rotate_image_clockwise(image: np.ndarray) -> np.ndarray:
    """Rotate an image clockwise and return a CV-friendly contiguous array."""

    return np.ascontiguousarray(np.rot90(image, k=-1))


@dataclass(frozen=True)
class ProjectionMetrics:
    total_ms: float
    projection_ms: float
    z_buffer_ms: float
    valid_points: int
    projected_points: int


class DepthColorProjector:
    """Build depth/color pixel correspondences in the calibration's coordinates."""

    def __init__(
        self,
        depth_intrinsics: CameraIntrinsics,
        color_intrinsics: CameraIntrinsics,
        depth_to_color: RigidTransform,
    ) -> None:
        self.depth_intrinsics = depth_intrinsics
        self.color_intrinsics = color_intrinsics
        self.rotation = np.asarray(depth_to_color.rotation, dtype=np.float32)
        self.translation = np.asarray(depth_to_color.translation, dtype=np.float32)
        self.depth_to_color_x = np.empty((0, 0), dtype=np.int32)
        self.depth_to_color_y = np.empty((0, 0), dtype=np.int32)
        self.color_to_depth = np.empty((0, 0), dtype=np.int32)
        self._depth_shape = (0, 0)
        self._ray_x = np.empty((0, 0), dtype=np.float32)
        self._ray_y = np.empty((0, 0), dtype=np.float32)

    @staticmethod
    def _scaled_parameters(
        intrinsics: CameraIntrinsics,
        shape: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        height, width = shape
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid image shape: {shape}")
        calibrated_aspect = intrinsics.width / intrinsics.height
        actual_aspect = width / height
        if not np.isclose(actual_aspect, calibrated_aspect, rtol=1e-3):
            raise ValueError(
                f"image shape {shape} is incompatible with calibrated size "
                f"({intrinsics.height}, {intrinsics.width}); check the rotation setting"
            )
        scale_x = width / intrinsics.width
        scale_y = height / intrinsics.height
        fx = intrinsics.fx * scale_x
        fy = intrinsics.fy * scale_y
        cx = (intrinsics.cx + 0.5) * scale_x - 0.5
        cy = (intrinsics.cy + 0.5) * scale_y - 0.5
        return fx, fy, cx, cy

    def _prepare_depth_rays(self, shape: tuple[int, int]) -> None:
        if shape == self._depth_shape:
            return
        height, width = shape
        fx, fy, cx, cy = self._scaled_parameters(self.depth_intrinsics, shape)
        pixel_x, pixel_y = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        self._ray_x = (pixel_x - cx) / fx
        self._ray_y = (pixel_y - cy) / fy
        self._depth_shape = shape

    def build(
        self,
        depth_meters: np.ndarray,
        color_shape: tuple[int, int],
    ) -> ProjectionMetrics:
        """Build forward and z-buffered inverse lookup maps for one frame."""

        started = perf_counter()
        depth_shape = depth_meters.shape
        self._prepare_depth_rays(depth_shape)
        color_height, color_width = color_shape
        color_fx, color_fy, color_cx, color_cy = self._scaled_parameters(
            self.color_intrinsics, color_shape
        )

        depth = np.asarray(depth_meters, dtype=np.float32)
        depth_flat = depth.ravel()
        valid_indices = np.flatnonzero(np.isfinite(depth_flat) & (depth_flat > 0))
        z_depth = depth_flat[valid_indices]
        x_depth = self._ray_x.ravel()[valid_indices] * z_depth
        y_depth = self._ray_y.ravel()[valid_indices] * z_depth

        points_depth = np.column_stack((x_depth, y_depth, z_depth))
        points_color = points_depth @ self.rotation.T + self.translation
        z_color = points_color[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            projected_x = np.rint(color_fx * points_color[:, 0] / z_color + color_cx)
            projected_y = np.rint(color_fy * points_color[:, 1] / z_color + color_cy)

        inside = (
            np.isfinite(projected_x)
            & np.isfinite(projected_y)
            & (z_color > 0)
            & (projected_x >= 0)
            & (projected_x < color_width)
            & (projected_y >= 0)
            & (projected_y < color_height)
        )
        source_indices = valid_indices[inside]
        color_x = projected_x[inside].astype(np.int32)
        color_y = projected_y[inside].astype(np.int32)
        visible_z = z_color[inside]

        self.depth_to_color_x = np.full(depth_shape, -1, dtype=np.int32)
        self.depth_to_color_y = np.full(depth_shape, -1, dtype=np.int32)
        self.depth_to_color_x.ravel()[source_indices] = color_x
        self.depth_to_color_y.ravel()[source_indices] = color_y
        projected_at = perf_counter()

        destination_indices = color_y * color_width + color_x
        z_buffer = np.full(color_height * color_width, np.inf, dtype=np.float32)
        np.minimum.at(z_buffer, destination_indices, visible_z)
        frontmost = visible_z <= z_buffer[destination_indices] + 1e-6
        self.color_to_depth = np.full(color_shape, -1, dtype=np.int32)
        self.color_to_depth.ravel()[destination_indices[frontmost]] = source_indices[frontmost]
        finished = perf_counter()

        return ProjectionMetrics(
            total_ms=(finished - started) * 1000,
            projection_ms=(projected_at - started) * 1000,
            z_buffer_ms=(finished - projected_at) * 1000,
            valid_points=len(valid_indices),
            projected_points=len(source_indices),
        )

    def projected_color_pixel(self, depth_x: int, depth_y: int) -> tuple[int, int] | None:
        """Return the projected color pixel for one depth pixel."""

        color_x = int(self.depth_to_color_x[depth_y, depth_x])
        color_y = int(self.depth_to_color_y[depth_y, depth_x])
        return None if color_x < 0 or color_y < 0 else (color_x, color_y)

    def nearest_depth_pixel(
        self,
        color_x: int,
        color_y: int,
        *,
        radius: int = 3,
    ) -> tuple[int, int] | None:
        """Find the nearest projected depth sample around a color pixel."""

        height, width = self.color_to_depth.shape
        x0, x1 = max(0, color_x - radius), min(width, color_x + radius + 1)
        y0, y1 = max(0, color_y - radius), min(height, color_y + radius + 1)
        window = self.color_to_depth[y0:y1, x0:x1]
        candidate_y, candidate_x = np.nonzero(window >= 0)
        if not len(candidate_x):
            return None
        distances = (candidate_x + x0 - color_x) ** 2 + (candidate_y + y0 - color_y) ** 2
        nearest = int(np.argmin(distances))
        depth_index = int(window[candidate_y[nearest], candidate_x[nearest]])
        depth_width = self.depth_to_color_x.shape[1]
        return depth_index % depth_width, depth_index // depth_width


@dataclass(frozen=True)
class RGBDFrame:
    """One paired RGB-D sample from the recording."""

    index: int
    timestamp_ns: int
    color_timestamp_ns: int | None
    depth_timestamp_ns: int | None
    color: np.ndarray | None
    depth: np.ndarray | None
    depth_units: float
    color_metadata: dict[str, str]
    depth_metadata: dict[str, str]

    @property
    def depth_meters(self) -> np.ndarray | None:
        """Depth as float32 meters, or ``None`` if no depth frame was paired."""

        if self.depth is None:
            return None
        return self.depth.astype(np.float32) * self.depth_units


class RGBDDB3Reader:
    """Iterate color/depth frames from a RealSense-style ROS 2 sqlite bag."""

    def __init__(
        self,
        path: str | Path,
        *,
        color_topic: str | None = None,
        depth_topic: str | None = None,
        max_pair_delta_ns: int | None = 50_000_000,
        copy_arrays: bool = True,
        rotate_clockwise: bool = False,
    ) -> None:
        self.path = Path(path)
        self.max_pair_delta_ns = max_pair_delta_ns
        self.copy_arrays = copy_arrays
        self.rotate_clockwise = rotate_clockwise
        self._typestore = get_typestore(Stores.ROS2_HUMBLE)
        self._conn = sqlite3.connect(self.path)
        self._cursor = self._conn.cursor()
        self._topics = self._load_topics()
        self.color_topic = self._resolve_topic(color_topic, "/Color_0/image/data", required=False)
        self.depth_topic = self._resolve_topic(depth_topic, "/Depth_0/image/data", required=True)
        if self.depth_topic is None:
            raise ValueError(f"no depth topic found in {self.path}")
        self.color_metadata_topic = self._metadata_topic_for(self.color_topic)
        self.depth_metadata_topic = self._metadata_topic_for(self.depth_topic)
        self.native_color_intrinsics = self._read_camera_intrinsics(self.color_topic)
        self.native_depth_intrinsics = self._read_camera_intrinsics(self.depth_topic)
        self.native_depth_to_color = self._read_reference_transform(self.color_topic)
        if rotate_clockwise:
            self.color_intrinsics = (
                rotate_intrinsics_clockwise(self.native_color_intrinsics)
                if self.native_color_intrinsics is not None
                else None
            )
            self.depth_intrinsics = (
                rotate_intrinsics_clockwise(self.native_depth_intrinsics)
                if self.native_depth_intrinsics is not None
                else None
            )
            self.depth_to_color = (
                rotate_transform_clockwise(self.native_depth_to_color)
                if self.native_depth_to_color is not None
                else None
            )
        else:
            self.color_intrinsics = self.native_color_intrinsics
            self.depth_intrinsics = self.native_depth_intrinsics
            self.depth_to_color = self.native_depth_to_color
        self.depth_units = self._read_depth_units()
        self._color_rows = self._message_index(self.color_topic) if self.color_topic else []
        self._depth_rows = self._message_index(self.depth_topic)
        self._color_metadata_rows = (
            self._message_index(self.color_metadata_topic) if self.color_metadata_topic else []
        )
        self._depth_metadata_rows = (
            self._message_index(self.depth_metadata_topic) if self.depth_metadata_topic else []
        )
        self._pairs = self._build_pairs()
        self._position = 0

    def __enter__(self) -> "RGBDDB3Reader":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()

    def __iter__(self) -> Iterator[RGBDFrame]:
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield frame

    def __len__(self) -> int:
        return len(self._pairs)

    @property
    def fps(self) -> float | None:
        """Approximate FPS of the paired stream."""

        if len(self._pairs) < 2:
            return None
        start = self._pairs[0][0]
        end = self._pairs[-1][0]
        seconds = (end - start) / 1_000_000_000
        if seconds <= 0:
            return None
        return (len(self._pairs) - 1) / seconds

    def make_projector(self) -> DepthColorProjector:
        """Create a projector using the reader's active orientation."""

        if (
            self.depth_intrinsics is None
            or self.color_intrinsics is None
            or self.depth_to_color is None
        ):
            raise ValueError("the recording does not contain complete RGB-D calibration data")
        return DepthColorProjector(
            self.depth_intrinsics,
            self.color_intrinsics,
            self.depth_to_color,
        )

    def read(self) -> tuple[bool, RGBDFrame | None]:
        """Return ``(True, frame)`` or ``(False, None)`` when exhausted."""

        if self._position >= len(self._pairs):
            return False, None
        frame = self._read_pair(self._position, self._pairs[self._position])
        self._position += 1
        return True, frame

    def seek(self, index: int) -> None:
        """Move the next ``read()`` call to a paired-frame index."""

        if index < 0 or index > len(self._pairs):
            raise IndexError(f"frame index out of range: {index}")
        self._position = index

    def release(self) -> None:
        """Close the underlying sqlite database."""

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _load_topics(self) -> dict[str, tuple[int, str]]:
        rows = self._cursor.execute("select id, name, type from topics").fetchall()
        return {name: (topic_id, msg_type) for topic_id, name, msg_type in rows}

    def _resolve_topic(self, requested: str | None, suffix: str, *, required: bool) -> str | None:
        if requested is not None:
            if requested not in self._topics:
                raise KeyError(f"topic not found in {self.path}: {requested}")
            return requested

        candidates = [
            name
            for name, (_, msg_type) in self._topics.items()
            if name.endswith(suffix) and msg_type == "sensor_msgs/msg/Image"
        ]
        if len(candidates) != 1:
            if not required and not candidates:
                return None
            raise ValueError(
                f"expected one topic ending in {suffix!r}, found {len(candidates)}: {candidates}"
            )
        return candidates[0]

    def _metadata_topic_for(self, image_topic: str | None) -> str | None:
        if image_topic is None:
            return None
        topic = image_topic.removesuffix("/data") + "/metadata"
        return topic if topic in self._topics else None

    def _read_camera_intrinsics(self, image_topic: str | None) -> CameraIntrinsics | None:
        if image_topic is None:
            return None
        topic = image_topic.removesuffix("/image/data") + "/camera_info"
        if topic not in self._topics:
            return None

        text = self._read_string_by_topic(topic)
        if text is None:
            return None
        values = parse_metadata(text)
        required = {"width", "height", "fx", "fy", "ppx", "ppy"}
        if not required.issubset(values):
            return None
        coefficients = tuple(
            float(value) for value in values.get("coeffs", "").split(",") if value
        )
        return CameraIntrinsics(
            width=int(values["width"]),
            height=int(values["height"]),
            fx=float(values["fx"]),
            fy=float(values["fy"]),
            cx=float(values["ppx"]),
            cy=float(values["ppy"]),
            distortion_model=values.get("model", ""),
            distortion_coefficients=coefficients,
        )

    def _read_reference_transform(self, image_topic: str | None) -> RigidTransform | None:
        if image_topic is None:
            return None
        topic = image_topic.removesuffix("/image/data") + "/tf/ref_0"
        if topic not in self._topics:
            return None

        text = self._read_string_by_topic(topic)
        if text is None:
            return None
        values = parse_metadata(text)
        rotation_values = [
            float(value) for value in values.get("rotation", "").split(",") if value
        ]
        translation = tuple(
            float(value) for value in values.get("translation", "").split(",") if value
        )
        if len(rotation_values) != 9 or len(translation) != 3:
            return None
        rotation = tuple(
            tuple(rotation_values[row * 3 : (row + 1) * 3]) for row in range(3)
        )
        return RigidTransform(
            rotation=rotation,
            translation=cast(tuple[float, float, float], translation),
        )

    def _read_depth_units(self) -> float:
        option_topic = "/device_0/sensor_0/option/Depth_Units/value"
        if option_topic in self._topics:
            message = self._read_string_by_topic(option_topic)
            if message:
                return float(message)

        if self.depth_metadata_topic is not None:
            rows = self._message_index(self.depth_metadata_topic)
            if rows:
                metadata_msg = cast(StringMessage, self._deserialize(rows[0][1], self.depth_metadata_topic))
                metadata = parse_metadata(metadata_msg.data)
                if "depth_units" in metadata:
                    return float(metadata["depth_units"])

        return 0.001

    def _read_string_by_topic(self, topic: str) -> str | None:
        topic_id, _ = self._topics[topic]
        row = self._cursor.execute(
            "select data from messages where topic_id = ? order by timestamp limit 1",
            (topic_id,),
        ).fetchone()
        if row is None:
            return None
        msg = cast(StringMessage, self._deserialize_payload(row[0], self._topics[topic][1]))
        return msg.data

    def _message_index(self, topic: str) -> list[tuple[int, int]]:
        topic_id, _ = self._topics[topic]
        return self._cursor.execute(
            "select timestamp, id from messages where topic_id = ? order by timestamp",
            (topic_id,),
        ).fetchall()

    def _build_pairs(self) -> list[tuple[int, int | None, int | None]]:
        if not self._depth_rows:
            raise ValueError(f"no depth frames found on {self.depth_topic}")

        color_timestamps = [timestamp for timestamp, _ in self._color_rows]
        pairs: list[tuple[int, int | None, int | None]] = []
        for depth_timestamp, depth_message_id in self._depth_rows:
            color_message_id = self._nearest_message_id(color_timestamps, depth_timestamp)
            pairs.append((depth_timestamp, color_message_id, depth_message_id))
        return pairs

    def _nearest_message_id(self, timestamps: list[int], target: int) -> int | None:
        if not timestamps:
            return None

        insert_at = bisect_left(timestamps, target)
        candidates = []
        if insert_at > 0:
            candidates.append(self._color_rows[insert_at - 1])
        if insert_at < len(self._color_rows):
            candidates.append(self._color_rows[insert_at])

        nearest_timestamp, nearest_message_id = min(
            candidates,
            key=lambda row: abs(row[0] - target),
        )
        if (
            self.max_pair_delta_ns is not None
            and abs(nearest_timestamp - target) > self.max_pair_delta_ns
        ):
            return None
        return nearest_message_id

    def _read_pair(
        self,
        index: int,
        pair: tuple[int, int | None, int | None],
    ) -> RGBDFrame:
        timestamp_ns, color_message_id, depth_message_id = pair
        color_msg = (
            cast(ImageMessage, self._deserialize(color_message_id, self.color_topic))
            if color_message_id
            else None
        )
        depth_msg = (
            cast(ImageMessage, self._deserialize(depth_message_id, self.depth_topic))
            if depth_message_id
            else None
        )
        color = image_to_array(color_msg, copy=self.copy_arrays) if color_msg else None
        depth = image_to_array(depth_msg, copy=self.copy_arrays) if depth_msg else None
        if self.rotate_clockwise:
            color = rotate_image_clockwise(color) if color is not None else None
            depth = rotate_image_clockwise(depth) if depth is not None else None

        return RGBDFrame(
            index=index,
            timestamp_ns=timestamp_ns,
            color_timestamp_ns=self._timestamp_for_id(color_message_id),
            depth_timestamp_ns=self._timestamp_for_id(depth_message_id),
            color=color,
            depth=depth,
            depth_units=self.depth_units,
            color_metadata=self._metadata_for_message(color_message_id, self._color_metadata_rows),
            depth_metadata=self._metadata_for_message(depth_message_id, self._depth_metadata_rows),
        )

    def _metadata_for_message(
        self,
        image_message_id: int | None,
        metadata_rows: list[tuple[int, int]],
    ) -> dict[str, str]:
        if image_message_id is None or not metadata_rows:
            return {}

        image_timestamp = self._timestamp_for_id(image_message_id)
        if image_timestamp is None:
            return {}
        timestamps = [timestamp for timestamp, _ in metadata_rows]
        metadata_message_id = nearest_message_id(metadata_rows, timestamps, image_timestamp)
        if metadata_message_id is None:
            return {}
        metadata_msg = cast(
            StringMessage,
            self._deserialize(metadata_message_id, self._topic_for_rows(metadata_rows)),
        )
        return parse_metadata(metadata_msg.data)

    def _deserialize(self, message_id: int, topic: str | None) -> Any:
        if topic is None:
            raise ValueError("cannot deserialize a message without a topic")
        row = self._cursor.execute(
            "select data from messages where id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"message id not found: {message_id}")
        return self._deserialize_payload(row[0], self._topics[topic][1])

    def _deserialize_payload(self, payload: bytes, msg_type: str) -> Any:
        return self._typestore.deserialize_cdr(payload, msg_type)

    def _topic_for_rows(self, rows: list[tuple[int, int]]) -> str:
        if rows is self._color_metadata_rows and self.color_metadata_topic is not None:
            return self.color_metadata_topic
        if rows is self._depth_metadata_rows and self.depth_metadata_topic is not None:
            return self.depth_metadata_topic
        raise ValueError("metadata rows do not match a known topic")

    def _timestamp_for_id(self, message_id: int | None) -> int | None:
        if message_id is None:
            return None
        row = self._cursor.execute(
            "select timestamp from messages where id = ?",
            (message_id,),
        ).fetchone()
        return int(row[0]) if row is not None else None


def image_to_array(message: ImageMessage, *, copy: bool = True) -> np.ndarray:
    """Convert a deserialized ``sensor_msgs/Image`` into a NumPy array."""

    encoding = message.encoding.lower()
    buffer = message.data.tobytes() if isinstance(message.data, np.ndarray) else bytes(message.data)

    if encoding in {"rgb8", "bgr8"}:
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(message.height, message.width, 3)
    elif encoding in {"rgba8", "bgra8"}:
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(message.height, message.width, 4)
    elif encoding in {"mono8", "8uc1"}:
        array = np.frombuffer(buffer, dtype=np.uint8).reshape(message.height, message.width)
    elif encoding in {"mono16", "16uc1"}:
        dtype = ">u2" if message.is_bigendian else "<u2"
        array = np.frombuffer(buffer, dtype=np.dtype(dtype)).reshape(message.height, message.width)
    else:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}")

    return array.copy() if copy else array


def parse_metadata(text: str) -> dict[str, str]:
    """Parse RealSense metadata strings like ``key=value;key=value;``."""

    metadata: dict[str, str] = {}
    for item in text.split(";"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def nearest_message_id(
    rows: list[tuple[int, int]],
    timestamps: list[int],
    target: int,
) -> int | None:
    if not rows:
        return None
    insert_at = bisect_left(timestamps, target)
    candidates = []
    if insert_at > 0:
        candidates.append(rows[insert_at - 1])
    if insert_at < len(rows):
        candidates.append(rows[insert_at])
    return min(candidates, key=lambda row: abs(row[0] - target))[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a RealSense-style RGB-D db3 file.")
    parser.add_argument("bag", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--rotate-clockwise", action="store_true")
    args = parser.parse_args()

    with RGBDDB3Reader(args.bag, rotate_clockwise=args.rotate_clockwise) as reader:
        print(f"frames={len(reader)} fps={reader.fps:.2f} depth_units={reader.depth_units}")
        print(f"color_topic={reader.color_topic}")
        print(f"depth_topic={reader.depth_topic}")
        for index, frame in zip(range(args.frames), reader):
            color_shape = None if frame.color is None else frame.color.shape
            depth_shape = None if frame.depth is None else frame.depth.shape
            depth_min = None if frame.depth is None else int(frame.depth.min())
            depth_max = None if frame.depth is None else int(frame.depth.max())
            print(
                f"{index}: color={color_shape} depth={depth_shape} "
                f"depth_range=({depth_min}, {depth_max})"
            )
