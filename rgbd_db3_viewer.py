"""Browse an RGB-D ``.db3`` recording using OpenCV windows.

Run with::

    python rgbd_db3_viewer.py recordings/recording.db3

Focus either image window and press ``n``, ``p``, or ``q``. A background worker
automatically builds the calibrated depth/color correspondence map for the
newest displayed frame.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rgbd_db3_reader import (
    DepthColorProjector,
    ProjectionMetrics,
    RigidTransform,
    RGBDDB3Reader,
    RGBDFrame,
)


COLOR_WINDOW = "RGB-D Viewer - Color"
DEPTH_WINDOW = "RGB-D Viewer - Depth"
WINDOW_NAMES = (COLOR_WINDOW, DEPTH_WINDOW)


@dataclass
class DisplayPane:
    """One native-resolution image and its scaled display representation."""

    name: str
    native_bgr: np.ndarray
    display_bgr: np.ndarray
    scale_x: float
    scale_y: float
    cursor_native: tuple[int, int] | None = None
    mouse_display: tuple[int, int] | None = None

    @classmethod
    def create(
        cls,
        name: str,
        native_bgr: np.ndarray,
        maximum_height: int,
    ) -> DisplayPane:
        native_height, native_width = native_bgr.shape[:2]
        display_height = min(native_height, maximum_height)
        scale = display_height / native_height
        display_width = max(1, round(native_width * scale))

        if (display_width, display_height) == (native_width, native_height):
            displayed = native_bgr.copy()
        else:
            displayed = cv2.resize(
                native_bgr,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA,
            )

        return cls(
            name=name,
            native_bgr=native_bgr,
            display_bgr=displayed,
            scale_x=display_width / native_width,
            scale_y=display_height / native_height,
        )

    def to_native(self, display_x: int, display_y: int) -> tuple[int, int]:
        """Convert display coordinates into clipped native-image coordinates."""

        height, width = self.native_bgr.shape[:2]
        native_x = int(np.clip(round(display_x / self.scale_x), 0, width - 1))
        native_y = int(np.clip(round(display_y / self.scale_y), 0, height - 1))
        return native_x, native_y

    def to_display(self, native_x: int, native_y: int) -> tuple[int, int]:
        """Convert native-image coordinates into clipped display coordinates."""

        height, width = self.display_bgr.shape[:2]
        display_x = int(np.clip(round(native_x * self.scale_x), 0, width - 1))
        display_y = int(np.clip(round(native_y * self.scale_y), 0, height - 1))
        return display_x, display_y


class OpenCVRGBDViewer:
    """Render linked color and depth views using OpenCV HighGUI."""

    def __init__(
        self,
        projector: DepthColorProjector,
        *,
        display_height: int = 800,
        depth_min_m: float = 0.25,
        depth_max_m: float = 2.0,
    ) -> None:
        if display_height <= 0:
            raise ValueError("display_height must be positive")
        if depth_max_m <= depth_min_m:
            raise ValueError("depth_max_m must be greater than depth_min_m")

        self.projector = projector
        self.display_height = display_height
        self.depth_min_m = depth_min_m
        self.depth_max_m = depth_max_m
        self.frame: RGBDFrame | None = None
        self.depth_meters: np.ndarray | None = None
        self.panes: dict[str, DisplayPane] = {}
        self.metrics: ProjectionMetrics | None = None
        self._active_window: str | None = None
        self._windows_created = False
        self._generation = 0
        self._map_future: Future[
            tuple[int, DepthColorProjector, ProjectionMetrics]
        ] | None = None
        self._map_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rgbd-map",
        )

    @property
    def mapping_ready(self) -> bool:
        """Whether correspondence data exists for the displayed frame."""

        return self.metrics is not None

    @property
    def mapping_in_progress(self) -> bool:
        """Whether the current frame is waiting for its background map."""

        return self._map_future is not None and not self._map_future.done()

    def create_windows(self) -> None:
        """Create both OpenCV windows and register their mouse callbacks."""

        for window_name in WINDOW_NAMES:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_name, self._on_mouse, window_name)
        self._windows_created = True

    def set_frame(self, frame: RGBDFrame) -> None:
        """Display a new frame and invalidate the previous correspondence map."""

        self.frame = frame
        self.depth_meters = frame.depth_meters
        self.metrics = None
        self._generation += 1
        self._active_window = None
        self.panes = {
            COLOR_WINDOW: DisplayPane.create(
                COLOR_WINDOW,
                self._color_visualization(frame),
                self.display_height,
            ),
            DEPTH_WINDOW: DisplayPane.create(
                DEPTH_WINDOW,
                self._depth_visualization(self.depth_meters),
                self.display_height,
            ),
        }
        self._schedule_map()

    def poll_map(self) -> ProjectionMetrics | None:
        """Adopt a completed background map if it belongs to this frame."""

        future = self._map_future
        if future is None or not future.done():
            return None
        self._map_future = None

        generation, projector, metrics = future.result()
        if generation != self._generation:
            return None

        self.projector = projector
        self.metrics = metrics
        self._update_linked_cursors()
        if self._windows_created:
            self.show()
        return metrics

    def close(self) -> None:
        """Stop accepting map work and finish any running projection safely."""

        if self._map_future is not None:
            self._map_future.cancel()
            self._map_future = None
        self._map_executor.shutdown(wait=True, cancel_futures=True)

    def _schedule_map(self) -> None:
        """Queue only the newest frame; obsolete queued work is cancelled."""

        if self._map_future is not None:
            self._map_future.cancel()
            self._map_future = None
        if self.frame is None or self.frame.color is None or self.depth_meters is None:
            return

        generation = self._generation
        projector = self._new_projector()
        depth_meters = self.depth_meters
        color_shape = self.frame.color.shape[:2]
        self._map_future = self._map_executor.submit(
            self._build_map,
            generation,
            projector,
            depth_meters,
            color_shape,
        )

    def _new_projector(self) -> DepthColorProjector:
        """Make an isolated projector whose arrays belong to one worker job."""

        transform = RigidTransform(
            rotation=tuple(
                tuple(float(value) for value in row) for row in self.projector.rotation
            ),
            translation=tuple(float(value) for value in self.projector.translation),
        )
        return DepthColorProjector(
            self.projector.depth_intrinsics,
            self.projector.color_intrinsics,
            transform,
        )

    @staticmethod
    def _build_map(
        generation: int,
        projector: DepthColorProjector,
        depth_meters: np.ndarray,
        color_shape: tuple[int, int],
    ) -> tuple[int, DepthColorProjector, ProjectionMetrics]:
        metrics = projector.build(depth_meters, color_shape)
        return generation, projector, metrics

    def show(self) -> None:
        """Redraw both windows from clean images and the current hover state."""

        for window_name in WINDOW_NAMES:
            cv2.imshow(window_name, self.render(window_name))

    def render(self, window_name: str) -> np.ndarray:
        """Return one fully annotated display image."""

        pane = self.panes[window_name]
        rendered = pane.display_bgr.copy()
        if pane.cursor_native is not None:
            self._draw_crosshair(rendered, pane.to_display(*pane.cursor_native))
        self._draw_annotation(rendered, pane)
        return rendered

    def windows_open(self) -> bool:
        """Return false if either OpenCV window has been closed."""

        if not self._windows_created:
            return False
        try:
            return all(
                cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
                for name in WINDOW_NAMES
            )
        except cv2.error:
            # Some HighGUI backends remove a closed window immediately instead
            # of retaining it with a zero-valued visibility property.
            return False

    def _on_mouse(
        self,
        event: int,
        display_x: int,
        display_y: int,
        _flags: int,
        window_name: str,
    ) -> None:
        if event != cv2.EVENT_MOUSEMOVE or window_name not in self.panes:
            return

        pane = self.panes[window_name]
        pane.mouse_display = (display_x, display_y)
        pane.cursor_native = pane.to_native(display_x, display_y)
        self._active_window = window_name
        self._update_linked_cursors()
        if self._windows_created:
            self.show()

    def _update_linked_cursors(self) -> None:
        if self._active_window not in self.panes:
            return

        source = self.panes[self._active_window]
        target_name = (
            DEPTH_WINDOW if self._active_window == COLOR_WINDOW else COLOR_WINDOW
        )
        target = self.panes[target_name]
        target.mouse_display = None
        target.cursor_native = None

        if not self.mapping_ready or source.cursor_native is None:
            return

        source_x, source_y = source.cursor_native
        if self._active_window == DEPTH_WINDOW:
            target.cursor_native = self.projector.projected_color_pixel(
                source_x,
                source_y,
            )
        else:
            target.cursor_native = self.projector.nearest_depth_pixel(
                source_x,
                source_y,
            )

    @staticmethod
    def _draw_crosshair(image: np.ndarray, point: tuple[int, int]) -> None:
        x, y = point
        color = (255, 0, 255)
        arm = 10
        gap = 3
        thickness = 1
        cv2.line(image, (x - arm, y), (x - gap, y), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x + gap, y), (x + arm, y), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x, y - arm), (x, y - gap), color, thickness, cv2.LINE_AA)
        cv2.line(image, (x, y + gap), (x, y + arm), color, thickness, cv2.LINE_AA)

    def _draw_annotation(self, image: np.ndarray, pane: DisplayPane) -> None:
        assert self.frame is not None

        if self.metrics is not None:
            map_text = f"map {self.metrics.total_ms:.1f} ms"
        elif self.mapping_in_progress:
            map_text = "building map..."
        else:
            map_text = "map unavailable"
        lines = [
            f"frame {self.frame.index}  t={self.frame.timestamp_ns / 1e9:.3f}s",
            map_text,
        ]
        if pane.cursor_native is not None:
            x, y = pane.cursor_native
            lines.append(f"pixel ({x}, {y})")
            if pane.name == DEPTH_WINDOW and self.depth_meters is not None:
                depth = float(self.depth_meters[y, x])
                lines.append(f"depth {depth:.3f} m" if depth > 0 else "depth invalid")
        lines.append("n next | p previous | q quit")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 1
        padding = 8
        line_gap = 5
        text_sizes = [
            cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines
        ]
        line_height = max(height for _, height in text_sizes)
        box_width = max(width for width, _ in text_sizes) + 2 * padding
        box_height = (
            len(lines) * line_height + (len(lines) - 1) * line_gap + 2 * padding
        )

        image_height, image_width = image.shape[:2]
        margin = 8
        top_left = (margin, margin)
        bottom_right = (
            max(margin, image_width - margin - box_width),
            max(margin, image_height - margin - box_height),
        )

        # The label lives in the top-left until the pointer enters it. It then
        # moves to the diagonally opposite corner, leaving the hovered area clear.
        origin_x, origin_y = top_left
        if pane.mouse_display is not None:
            mouse_x, mouse_y = pane.mouse_display
            over_top_left = (
                origin_x <= mouse_x <= origin_x + box_width
                and origin_y <= mouse_y <= origin_y + box_height
            )
            if over_top_left:
                origin_x, origin_y = bottom_right

        cv2.rectangle(
            image,
            (origin_x, origin_y),
            (origin_x + box_width, origin_y + box_height),
            (0, 0, 0),
            cv2.FILLED,
        )
        baseline_y = origin_y + padding + line_height
        for line in lines:
            cv2.putText(
                image,
                line,
                (origin_x + padding, baseline_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            baseline_y += line_height + line_gap

    @staticmethod
    def _color_visualization(frame: RGBDFrame) -> np.ndarray:
        if frame.color is None:
            return OpenCVRGBDViewer._placeholder("No paired color frame")
        if frame.color.ndim == 2:
            return cv2.cvtColor(frame.color, cv2.COLOR_GRAY2BGR)
        if frame.color.shape[2] == 4:
            return cv2.cvtColor(frame.color, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)

    def _depth_visualization(self, depth_meters: np.ndarray | None) -> np.ndarray:
        if depth_meters is None:
            return self._placeholder("No depth frame")

        scaled = (depth_meters - self.depth_min_m) / (
            self.depth_max_m - self.depth_min_m
        )
        depth_u8 = np.clip(scaled * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        colored[~np.isfinite(depth_meters) | (depth_meters <= 0)] = 0
        return colored

    @staticmethod
    def _placeholder(message: str) -> np.ndarray:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            image,
            message,
            (30, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image


def print_projection_metrics(frame: RGBDFrame, metrics: ProjectionMetrics) -> None:
    """Print detailed timing for one background map build."""

    throughput = metrics.valid_points / metrics.total_ms / 1000
    print(
        f"frame {frame.index}: reprojection {metrics.total_ms:.2f} ms "
        f"(project {metrics.projection_ms:.2f} ms, "
        f"z-buffer {metrics.z_buffer_ms:.2f} ms, "
        f"{metrics.valid_points:,} valid, {metrics.projected_points:,} in frame, "
        f"{throughput:.2f} Mpoints/s)"
    )


def browse_recording(
    bag_path: Path,
    *,
    rotate_clockwise: bool = True,
    display_height: int = 800,
    depth_min_m: float = 0.25,
    depth_max_m: float = 2.0,
) -> None:
    """Open *bag_path* and browse paired frames using OpenCV key events."""

    with RGBDDB3Reader(bag_path, rotate_clockwise=rotate_clockwise) as reader:
        if len(reader) == 0:
            print("The recording contains no paired frames.")
            return

        fps_text = "unknown" if reader.fps is None else f"{reader.fps:.2f}"
        print(f"Loaded {len(reader)} frames (approximately {fps_text} FPS).")
        orientation = "90 degrees clockwise" if rotate_clockwise else "native"
        print(f"Processing orientation: {orientation}.")
        print("Focus either image window. Keys: n next, p previous, q quit.")
        print("Pixel correspondence maps are generated automatically in the background.")

        viewer = OpenCVRGBDViewer(
            reader.make_projector(),
            display_height=display_height,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        viewer.create_windows()
        index = 0
        timing_samples: list[float] = []

        try:
            while viewer.windows_open():
                reader.seek(index)
                ok, frame = reader.read()
                if not ok or frame is None:
                    break
                viewer.set_frame(frame)
                viewer.show()

                change_frame = False
                while viewer.windows_open() and not change_frame:
                    metrics = viewer.poll_map()
                    if metrics is not None:
                        timing_samples.append(metrics.total_ms)
                        print_projection_metrics(frame, metrics)

                    key_code = cv2.waitKeyEx(20)
                    if key_code < 0:
                        continue
                    key_value = key_code & 0xFF
                    if key_value > 0x7F:
                        continue
                    key = chr(key_value).lower()

                    if key == "q":
                        return
                    if key == "n":
                        if index == len(reader) - 1:
                            print("Already at the last frame.")
                            continue
                        index += 1
                        change_frame = True
                        continue
                    if key == "p":
                        if index == 0:
                            print("Already at the first frame.")
                            continue
                        index -= 1
                        change_frame = True
        finally:
            viewer.close()
            cv2.destroyAllWindows()
            if timing_samples:
                samples = np.asarray(timing_samples)
                print(
                    f"Reprojection summary ({len(samples)} builds): "
                    f"mean={samples.mean():.2f} ms, "
                    f"median={np.median(samples):.2f} ms, "
                    f"p95={np.percentile(samples, 95):.2f} ms"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse RGB-D frames with OpenCV.")
    parser.add_argument("bag", type=Path, help="path to the .db3 recording")
    parser.add_argument(
        "--native-orientation",
        action="store_true",
        help="do not rotate frames and calibration clockwise",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=800,
        help="maximum height of each display window (default: 800)",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=0.25,
        metavar="METERS",
        help="near end of the depth color scale (default: 0.25)",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=2.0,
        metavar="METERS",
        help="far end of the depth color scale (default: 2.0)",
    )
    args = parser.parse_args()
    browse_recording(
        args.bag,
        rotate_clockwise=not args.native_orientation,
        display_height=args.display_height,
        depth_min_m=args.depth_min,
        depth_max_m=args.depth_max,
    )


if __name__ == "__main__":
    main()
