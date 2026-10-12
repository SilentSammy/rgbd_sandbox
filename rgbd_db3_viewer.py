"""RGB-D viewer with hover inspection and shared keyboard playback controls."""

import cv2
import numpy as np
from time import perf_counter

from rgbd_projection import Projector


PLAYBACK_HELP = "A reverse | S pause | D play | Q previous | E next | W restart | Esc quit"


class Playback:
    """Start paused; step/restart pause, and playback stops at either end.

    Pace at the recording's FPS, without skipping frames when processing is slow.
    poll() also returns unhandled keys for each demo's own commands.
    """

    def __init__(self, source, start_frame=0):
        self.count = len(source)
        if not 0 <= start_frame < self.count:
            raise ValueError(f"start-frame must be between 0 and {self.count - 1}")
        self.index = self.start_frame = start_frame
        self.interval = 1 / (source.fps or 30)
        self.direction = 0
        self.quit = False
        self.next_frame = 0

    def seek(self, index):
        self.index = min(max(index, 0), self.count - 1)
        self.direction = 0

    def update(self, key=-1, *, now=None):
        now = perf_counter() if now is None else now
        command = chr(key & 0xFF).lower() if key >= 0 else ""
        if command == "\x1b":
            self.quit = True
        elif command in ("a", "d"):
            direction = -1 if command == "a" else 1
            if direction != self.direction:
                self.direction = direction
                self.next_frame = now + self.interval
        elif command == "s":
            self.direction = 0
        elif command in ("q", "e", "w"):
            self.seek(self.start_frame if command == "w" else
                      self.index + (-1 if command == "q" else 1))
        if not self.quit and self.direction and now >= self.next_frame:
            self.index = min(max(self.index + self.direction, 0), self.count - 1)
            self.next_frame = now + self.interval
            if self.index in (0, self.count - 1):
                self.direction = 0
        return command

    def poll(self):
        return self.update(cv2.waitKey(20))


def _resize(image, max_dimension):
    height, width = image.shape[:2]
    if max_dimension is None:
        return image
    scale = max_dimension / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, size, interpolation=interpolation)


def _native_pixel(x, y, display_shape, native_shape):
    """Map the center of a display pixel to the corresponding source pixel."""
    dh, dw = display_shape[:2]
    nh, nw = native_shape[:2]
    return min(nw - 1, int((x + 0.5) * nw / dw)), min(nh - 1, int((y + 0.5) * nh / dh))


def _display_pixel(u, v, display_shape, native_shape):
    dh, dw = display_shape[:2]
    nh, nw = native_shape[:2]
    x, y = round((u + 0.5) * dw / nw - 0.5), round((v + 0.5) * dh / nh - 0.5)
    return int(np.clip(x, 0, dw - 1)), int(np.clip(y, 0, dh - 1))


def _depth_pixel(inverse, u, v, radius=3):
    """Nearest measured inverse-map entry within radius RGB pixels, or None."""
    h,w = inverse.shape[:2]
    x0,y0 = max(0,u-radius),max(0,v-radius)
    patch = inverse[y0:min(h,v+radius+1),x0:min(w,u+radius+1)]
    ys,xs = np.nonzero(patch[:,:,0] >= 0)
    distance = (xs+x0-u)**2 + (ys+y0-v)**2
    within = distance <= radius**2
    if not np.any(within):
        return None
    nearest = np.argmin(np.where(within,distance,np.inf))
    return patch[ys[nearest],xs[nearest]], distance[nearest] != 0


def view(source, *, max_dimension=None, depth_range=(0.25, 2.0), start_frame=0):
    """Browse source[index] frames with RGB uint8 color and depth in meters.

    Uses Playback controls. Closing either window also exits.
    Hover marks only the other window. RGB lookup uses the nearest measured
    sample within 3 RGB pixels; approximate matches are labeled in the terminal.
    Native pixel coordinates/depth in meters are printed in the terminal.
    source.calibration supplies color/depth Cameras and a depth_to_color matrix.
    Display resizing and colorization do not modify the source arrays.
    max_dimension sets each image's longer axis in pixels, including upscaling.
    Leave it as None to display at native resolution.
    """
    count = len(source)
    if not count:
        print("No frames to display.")
        return
    near, far = depth_range
    if max_dimension is not None and (not isinstance(max_dimension, int) or max_dimension <= 0):
        raise ValueError("max_dimension must be a positive integer or None")
    if not np.isfinite([near, far]).all() or far <= near:
        raise ValueError("Require finite near < far")

    windows = ("RGB", "Depth")
    playback = Playback(source, start_frame)
    index, displayed = playback.index, None
    panes = {}
    mapping = None
    inverse = None
    projector = None
    calibration = source.calibration

    def on_mouse(event, x, y, flags, name):
        if event != cv2.EVENT_MOUSEMOVE or name not in panes:
            return
        image, native = panes[name]
        height, width = image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return
        u, v = _native_pixel(x, y, image.shape, native.shape)
        # Clear the previous match; the hovered window uses its native cursor.
        target = "RGB" if name == "Depth" else "Depth"
        target_image, target_native = panes[target]
        overlay = target_image.copy()
        corresponding = None
        text = f"Frame {index} | {name} pixel ({u}, {v})"
        if name == "Depth":
            meters = float(native[v, u])
            text += f" | {meters:.3f} m" if np.isfinite(meters) and meters > 0 else " | invalid depth"
            if np.isfinite(mapping[v,u]).all():
                corresponding = mapping[v,u]
        else:
            match = _depth_pixel(inverse,u,v)
            if match is not None:
                corresponding, approximate = match
                du,dv = corresponding
                text += f" | depth ({du}, {dv}): {target_native[dv,du]:.3f} m"
                if approximate:
                    text += " | nearby sample"
        if corresponding is not None:
            point = _display_pixel(*corresponding,target_image.shape,target_native.shape)
            cv2.drawMarker(overlay,point,(255,0,255),cv2.MARKER_CROSS,15,1)
        else:
            text += f" | no {target} match"
        cv2.imshow(name,image)  # Remove any marker from the hovered window.
        cv2.imshow(target,overlay)
        print(f"\r{text:<100}", end="", flush=True)

    try:
        print(PLAYBACK_HELP)
        for name in windows:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(name, on_mouse, name)
        while True:
            if index != displayed:
                frame = source[index]
                if projector is None or projector.depth_shape != frame.depth.shape or projector.color_shape != frame.color.shape[:2]:
                    projector = Projector(
                        calibration.depth.K,calibration.color.K,calibration.depth_to_color,
                        frame.depth.shape,frame.color.shape[:2],
                        depth_D=calibration.depth.D,color_D=calibration.color.D,
                        depth_model=calibration.depth.distortion_model,
                        color_model=calibration.color.distortion_model,
                    )
                started = perf_counter()
                mapping,inverse = projector.build_maps(frame.depth)
                map_ms = (perf_counter() - started) * 1000
                color = _resize(cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR), max_dimension)
                valid = np.isfinite(frame.depth) & (frame.depth > 0)
                values = np.where(valid, frame.depth, near)
                values = np.clip((values - near) / (far - near) * 255, 0, 255)
                depth = cv2.applyColorMap(values.astype(np.uint8), cv2.COLORMAP_TURBO)
                depth[~valid] = 0
                depth = _resize(depth, max_dimension)
                panes = {"RGB": (color, frame.color), "Depth": (depth, frame.depth)}
                cv2.imshow(windows[0], color)
                cv2.imshow(windows[1], depth)
                print(f"\r{f'Frame {index} | map {map_ms:.1f} ms | hover to inspect':<100}", end="", flush=True)
                if displayed is None:
                    cv2.moveWindow(windows[0], 20, 40)
                    cv2.moveWindow(windows[1], 40 + color.shape[1], 40)
                displayed = index

            playback.poll()
            try:
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows):
                    break
            except cv2.error:
                break  # A closed window may already have been removed by its backend.
            if playback.quit:
                break
            index = playback.index
    finally:
        print()
        for name in windows:
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                pass  # The user may already have closed this window.


if __name__ == "__main__":
    import argparse
    from rgbd_db3_reader import RealSenseReader

    parser = argparse.ArgumentParser(description=__doc__, epilog=PLAYBACK_HELP)
    parser.add_argument("bag", help="path to a RealSense Viewer .db3 recording")
    parser.add_argument("--rotations", type=int, choices=range(4), default=0,
                        help="clockwise quarter-turns: 0, 1, 2, or 3 (default: 0)")
    parser.add_argument("--max-dimension", type=int,
                        help="set each image's longer axis in pixels (default: native size)")
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    with RealSenseReader(args.bag, rotations=args.rotations) as recording:
        view(recording, max_dimension=args.max_dimension, start_frame=args.start_frame)
