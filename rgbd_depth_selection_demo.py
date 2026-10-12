"""Click an object to highlight its connected depth region on the current frame.

Shared viewer playback; c: clear, h: hull/raw, i/k: adjust depth jump.
Selections are replaced on each click and cleared on frame changes.
"""

import argparse
from time import perf_counter

import cv2
import numpy as np

from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _resize, _native_pixel, _depth_pixel
from rgbd_projection import Projector


def surface_normals(depth, camera):
    """Estimate local surface directions from lightly denoised metric depth."""
    z = cv2.medianBlur(np.where(np.isfinite(depth), depth, 0).astype(np.float32), 5)
    dx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8
    dy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8
    v, u = np.indices(depth.shape, dtype=np.float32)
    K = camera.K.astype(np.float32)
    normals = np.stack((-K[0, 0]*dx, -K[1, 1]*dy,
                        z + (u-K[0, 2])*dx + (v-K[1, 2])*dy), axis=2)
    return -normals / np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-9)


def surface_connections(depth, camera, concave_angle=6):
    """Allow flat/convex neighbors; reject concave folds using camera-facing normals.

    The tolerance is the difference of normal-to-connection angles, in degrees.
    Geometry is cached per frame; this is a pixel-level local-convexity heuristic.
    """
    valid = np.isfinite(depth) & (depth > 0)
    # Ignore missing samples while smoothing sensor noise over a small patch.
    z = cv2.GaussianBlur(np.where(valid, depth, 0), (0, 0), 4)
    z /= np.maximum(cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 4), 1e-9)
    normals = surface_normals(z, camera)
    v, u = np.indices(depth.shape, dtype=np.float32)
    K = camera.K.astype(np.float32)
    points = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)), axis=2)
    points *= z[..., None]

    def connected(p1, p2, n1, n2):
        delta = p2 - p1
        delta /= np.maximum(np.linalg.norm(delta, axis=2, keepdims=True), 1e-9)
        a1 = np.arccos(np.clip(np.sum(n1*delta, axis=2), -1, 1))
        a2 = np.arccos(np.clip(np.sum(n2*delta, axis=2), -1, 1))
        return a1 - a2 >= -np.deg2rad(concave_angle)

    connections = []
    for axis in (1, 0):
        i = np.arange(depth.shape[axis]-1)
        # Compare patches straddling each bond, not just consecutive noisy normals.
        a, b = np.maximum(i-5, 0), np.minimum(i+6, depth.shape[axis]-1)
        connections.append(connected(np.take(points, a, axis), np.take(points, b, axis),
                                     np.take(normals, a, axis), np.take(normals, b, axis)))
    return tuple(connections)


def merge_faces(labels, seed, points, depth, depth_jump):
    """Join nearby, nearly coplanar patches split by noisy local curvature.

    Whole-patch planes tolerate rippled box faces without relaxing every edge.
    This intentionally cannot distinguish adjacent coplanar, touching objects.
    """
    selected = labels == labels[seed[1], seed[0]]

    def plane(mask):
        cloud = points[mask]
        center = cloud.mean(axis=0)
        normal = np.linalg.svd(cloud-center, full_matrices=False)[2][-1]
        return center, normal

    # Tiny patches do not provide a dependable plane estimate.
    if selected.sum() < 50:
        return selected
    sizes = np.bincount(labels.ravel())
    fitted = {}
    for _ in range(3):
        center, normal = plane(selected)
        near = cv2.dilate(selected.astype(np.uint8), np.ones((31, 31), np.uint8)) != 0
        _, nearest = cv2.distanceTransformWithLabels((~selected).astype(np.uint8),
                            cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        closest = points[selected]
        coordinates = np.argwhere(selected)
        changed = False
        for label in np.unique(labels[near & ~selected]):
            if label == 0 or sizes[label] < 50:
                continue
            patch = labels == label
            gap = points[patch] - closest[nearest[patch]-1]
            distance = np.linalg.norm(gap, axis=1)
            best = np.argmin(distance)
            if distance[best] > 0.02:
                continue
            start = np.argwhere(patch)[best]
            end = coordinates[nearest[tuple(start)]-1]
            # A short gap must contain continuous measured depth, not background
            # or missing samples between two separate, parallel objects.
            line = np.rint(np.linspace(start, end, int(np.max(abs(end-start)))+1)).astype(int)
            values = depth[line[:, 0], line[:, 1]]
            if (not np.isfinite(values).all() or np.any(values <= 0) or
                    np.any(abs(np.diff(values)) > depth_jump)):
                continue
            if label not in fitted:
                fitted[label] = plane(patch)
            other, direction = fitted[label]
            offset = other-center
            if (abs(normal @ direction) >= np.cos(np.deg2rad(25)) and
                    max(abs(offset @ normal), abs(offset @ direction)) <= 0.02):
                y, x = line[len(line)//2]
                bridge = points[max(y-4, 0):y+5, max(x-4, 0):x+5].reshape(-1, 3)
                bridge = bridge[np.isfinite(bridge).all(axis=1) &
                                (abs(bridge[:, 2]-depth[y, x]) <= 0.02)]
                if len(bridge) < 10:
                    continue
                axis = np.linalg.svd(bridge-bridge.mean(axis=0), full_matrices=False)[2][-1]
                if min(abs(axis @ normal), abs(axis @ direction)) < np.cos(np.deg2rad(35)):
                    continue
                selected |= patch
                changed = True
        if not changed:
            break
    return selected


def select_region(depth, seed, depth_jump=0.01, seed_range=0.15, connections=None, camera=None):
    """Grow through 4-connected neighbors, stopping at depth jumps or missing data.

    depth_jump compares neighboring pixels, not every pixel against the seed.
    seed_range additionally limits total depth drift away from the clicked point.
    Both thresholds are in meters; gaps in measured depth are not filled.
    Connections additionally forbid crossing concave surface transitions.
    When camera is supplied, nearly coplanar fragments are reunited afterwards.
    """
    x, y = seed
    valid = np.isfinite(depth) & (depth > 0)
    if not valid[y, x]:
        return np.zeros(depth.shape, dtype=bool)
    allowed = valid & (np.abs(depth - depth[y, x]) <= seed_range)
    horizontal = allowed[:, :-1] & allowed[:, 1:] & (np.abs(np.diff(depth, axis=1)) <= depth_jump)
    vertical = allowed[:-1] & allowed[1:] & (np.abs(np.diff(depth, axis=0)) <= depth_jump)
    if connections is not None:
        horizontal &= connections[0]
        vertical &= connections[1]
    # Interleave pixels and edges so OpenCV can flood a graph with blocked bonds.
    graph = np.zeros((2*depth.shape[0]-1, 2*depth.shape[1]-1), dtype=np.uint8)
    graph[::2, ::2] = allowed
    graph[::2, 1::2] = horizontal
    graph[1::2, ::2] = vertical
    if camera is not None:
        _, labels = cv2.connectedComponentsWithAlgorithm(graph, 4, cv2.CV_32S, cv2.CCL_SAUF)
        v, u = np.indices(depth.shape, dtype=np.float32)
        K = camera.K.astype(np.float32)
        points = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)), axis=2)
        points *= depth[..., None]
        return merge_faces(labels[::2, ::2], seed, points, depth, depth_jump)
    cv2.floodFill(graph, None, (2*int(x), 2*int(y)), 2, flags=4)
    return graph[::2, ::2] == 2


def convex_mask(mask):
    """Fill the visible silhouette; never use this display hull for region growth."""
    filled = mask.astype(np.uint8)
    points = cv2.findNonZero(filled)
    if points is not None:
        cv2.fillConvexPoly(filled, cv2.convexHull(points), 1)
    return filled != 0


def highlight(image, mask):
    image = image.copy()
    image[mask] = (0.5 * image[mask] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
    return image


def view(source, *, max_dimension=None, depth_jump=0.01, seed_range=0.15,
         concave_angle=6, initial_frame=0):
    if not np.isfinite([depth_jump, seed_range]).all() or min(depth_jump, seed_range) <= 0:
        raise ValueError("Depth thresholds must be finite and positive")
    if max_dimension is not None and max_dimension <= 0:
        raise ValueError("max_dimension must be positive")
    if not 0 <= initial_frame < len(source):
        raise ValueError("Initial frame is outside the recording")
    if not 0 <= concave_angle <= 180:
        raise ValueError("concave_angle must be in [0, 180]")
    calibration = source.calibration
    projector = Projector(
        calibration.depth.K, calibration.color.K, calibration.depth_to_color,
        (calibration.depth.height, calibration.depth.width),
        (calibration.color.height, calibration.color.width),
        depth_D=calibration.depth.D, color_D=calibration.color.D,
        depth_model=calibration.depth.distortion_model, color_model=calibration.color.distortion_model,
    )
    windows = ("RGB selection", "Depth selection")
    playback = Playback(source, initial_frame)
    index, displayed = playback.index, None
    frame = inverse = seed = connections = None
    bases, panes = {}, {}
    selected = None
    show_hull = True

    def draw():
        rgb_mask = np.zeros(frame.color.shape[:2], dtype=bool)
        valid = inverse[:, :, 0] >= 0
        x, y = inverse[valid].T
        rgb_mask[valid] = selected[y, x]
        # Fill small projection holes for display only; depth selection stays exact.
        rgb_mask = cv2.dilate(rgb_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
        text = f"Frame {index} | {selected.sum()} pixels | {'hull' if show_hull else 'raw'} | jump {depth_jump*1000:.1f} mm"
        for name, mask in zip(windows, (rgb_mask, selected)):
            if show_hull:
                mask = convex_mask(mask)
            image = _resize(highlight(bases[name], mask), max_dimension)
            cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            panes[name] = image
            cv2.imshow(name, image)

    def select():
        nonlocal selected
        started = perf_counter()
        selected = select_region(frame.depth, seed, depth_jump, seed_range, connections, calibration.depth)
        print(f"Frame {index}, depth pixel {seed}: {selected.sum():,} pixels selected "
              f"in {(perf_counter()-started)*1000:.1f} ms; jump {depth_jump*1000:.1f} mm")
        draw()

    def on_mouse(event, x, y, flags, name):
        nonlocal seed
        if event != cv2.EVENT_LBUTTONDOWN or name not in panes:
            return
        image = panes[name]
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            return
        u, v = _native_pixel(x, y, image.shape, bases[name].shape)
        if name == windows[0]:
            match = _depth_pixel(inverse, u, v)
            if match is None:
                print("No measured depth near this RGB pixel; click elsewhere.")
                return
            seed = tuple(match[0])
            if match[1]:
                print("Using a nearby measured depth sample.")
        else:
            seed = (u, v)
        select()

    try:
        for name in windows:
            cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(name, on_mouse, name)
        print(PLAYBACK_HELP)
        print("Click either image. C clear | H hull/raw | I/K depth jump")
        while True:
            if index != displayed:
                frame = source[index]
                _, inverse = projector.build_maps(frame.depth)
                started = perf_counter()
                connections = surface_connections(frame.depth, calibration.depth, concave_angle)
                print(f"Frame {index}: surface connections {(perf_counter()-started)*1000:.1f} ms")
                seed = None
                selected = np.zeros(frame.depth.shape, dtype=bool)
                valid = np.isfinite(frame.depth) & (frame.depth > 0)
                values = np.clip((np.where(valid, frame.depth, 0.25) - 0.25) / 1.75 * 255, 0, 255)
                depth = cv2.applyColorMap(values.astype(np.uint8), cv2.COLORMAP_TURBO)
                depth[~valid] = 0
                bases = {windows[0]: cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR), windows[1]: depth}
                draw()
                if displayed is None:
                    cv2.moveWindow(windows[0], 20, 40)
                    cv2.moveWindow(windows[1], 40 + panes[windows[0]].shape[1], 40)
                displayed = index
            command = playback.poll()
            try:
                if any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1 for name in windows):
                    break
            except cv2.error:
                break
            if playback.quit:
                break
            if command == "c":
                seed = None
                selected[:] = False
                draw()
            elif command == "h":
                show_hull = not show_hull
                draw()
            elif command in ("i", "k"):
                depth_jump = max(0.001, round(depth_jump + (0.001 if command == "i" else -0.001), 6))
                if seed is not None:
                    select()
                else:
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
    parser.add_argument("--rotations", type=int, choices=range(4), default=1)
    parser.add_argument("--max-dimension", type=int)
    parser.add_argument("--start-frame", "--frame", dest="start_frame", type=int, default=0)
    parser.add_argument("--depth-jump", type=float, default=0.01,
                        help="maximum neighbor depth difference in meters (default: 0.01)")
    parser.add_argument("--seed-range", type=float, default=0.15,
                        help="maximum total depth difference from seed in meters (default: 0.15)")
    parser.add_argument("--concave-angle", type=float, default=6,
                        help="tolerated concavity in degrees (default: 6; 180 disables)")
    args = parser.parse_args()
    with RealSenseReader(args.bag, rotations=args.rotations) as source:
        view(source, max_dimension=args.max_dimension, depth_jump=args.depth_jump,
             seed_range=args.seed_range, concave_angle=args.concave_angle, initial_frame=args.start_frame)
