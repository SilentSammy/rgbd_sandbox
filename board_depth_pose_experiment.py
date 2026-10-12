"""Compare current RGB, subpixel RGB, and depth-plane board poses; no smoothing.

python board_depth_pose_experiment.py
This experiment does not change BoardEstimator or any demo's pose selection.
Results go to exports/<bag-stem>_depth_pose_s<scale>_r<limit>.json. Changes measure
stability, not accuracy: the camera moves and this bag has no reference poses.
"""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from board_pose_setup import BOARD, make_estimator
from rgbd_db3_reader import RealSenseReader
from rgbd_projection import Projector


def fit_plane(points):
    """Trim depth outliers and fit a plane; return normal, offset, retained mask."""
    points = np.asarray(points, dtype=float)
    keep = np.ones(len(points), dtype=bool)
    if len(points) < 100:
        raise ValueError("Too few board depth samples")
    # A larger patch can include shelf edges: initialize from a consensus plane
    # rather than letting those points tilt the first least-squares fit.
    triples = points[np.random.default_rng(0).integers(len(points), size=(128, 3))]
    normals = np.cross(triples[:, 1]-triples[:, 0], triples[:, 2]-triples[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-6
    if not np.any(usable):
        raise ValueError("Depth patch has no spatially distributed plane samples")
    normals = normals[usable] / lengths[usable, None]
    offsets = np.sum(normals * triples[usable, 0], axis=1)
    consensus = np.abs(points @ normals.T - offsets) < 0.008
    keep = consensus[:, np.argmax(consensus.sum(axis=0))]
    for _ in range(4):
        if keep.sum() < 100:
            raise ValueError("Too few depth inliers")
        center = points[keep].mean(axis=0)
        _, _, axes = np.linalg.svd(points[keep] - center, full_matrices=False)
        normal = axes[-1]
        distance = np.abs((points-center) @ normal)
        median = np.median(distance[keep])
        mad = np.median(np.abs(distance[keep]-median))
        keep = distance <= max(0.003, median + 3*1.4826*mad)
    if keep.sum() < 100:
        raise ValueError("Too few depth inliers")
    center = points[keep].mean(axis=0)
    _, _, axes = np.linalg.svd(points[keep]-center, full_matrices=False)
    normal = axes[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, float(center @ normal), keep


def rigid_pose(objects, targets):
    """Least-squares rigid alignment of known board points to measured 3D points."""
    a, b = objects.mean(axis=0), targets.mean(axis=0)
    u, _, vt = np.linalg.svd((objects-a).T @ (targets-b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return cv2.Rodrigues(rotation)[0].ravel(), b - rotation @ a


def depth_pose(frame, camera, pnp, projector, *, plane_scale=1.0, max_plane_rms=0.005):
    """Use measured board depth to constrain tilt/distance; RGB retains yaw/XY.

    Depth points are mapped into the existing estimator's reflected-pixel PnP
    convention before fitting, exactly as the point-memory demo does. No new
    image rotation or change to the public board axes is introduced.
    """
    started = perf_counter()
    mapping = projector.build_map(frame.depth)
    hull = cv2.convexHull(pnp.original_img_pts.astype(np.float32))
    center = hull.mean(axis=0)
    hull = center + plane_scale*(hull-center)
    mask = np.zeros(frame.color.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    samples = np.isfinite(mapping[:, :, 0])
    # Subsample spatially, not by selecting only the first part of the board.
    samples[1::2, :] = False
    samples[:, 1::2] = False
    y, x = np.nonzero(samples)
    pixels = mapping[y, x]
    u, v = np.rint(pixels).astype(int).T
    inside = mask[v, u] != 0
    pixels, y, x = pixels[inside], y[inside], x[inside]
    z = projector.rays[2][y, x] * frame.depth[y, x] + projector.translation[2]
    pixels = pnp.to_image_points(pixels)
    xy = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera.K, camera.D).reshape(-1, 2)
    points = np.column_stack((xy, np.ones(len(xy)))) * z[:, None]
    normal, offset, keep = fit_plane(points)
    residual = (points[keep] @ normal - offset)
    rms = float(np.sqrt(np.mean(residual**2)))
    if keep.mean() < 0.7 or rms > max_plane_rms:
        raise ValueError(f"Board depth plane rejected: {keep.mean():.0%} inliers, RMS {rms*1000:.1f} mm")
    rays = cv2.undistortPoints(pnp.img_pts.reshape(-1, 1, 2), camera.K, camera.D).reshape(-1, 2)
    rays = np.column_stack((rays, np.ones(len(rays))))
    targets = rays * (offset / (rays @ normal))[:, None]
    objects = np.column_stack((pnp.obj_pts, np.zeros(len(pnp.obj_pts))))
    rvec, tvec = rigid_pose(objects, targets)
    stats = {"samples": len(points), "inlier_fraction": float(keep.mean()),
             "plane_rms_mm": rms*1000, "depth_ms": (perf_counter()-started)*1000}
    return rvec, tvec, stats


def pose_record(pnp, camera, rvec=None, tvec=None):
    rvec = pnp.rvec if rvec is None else rvec
    tvec = pnp.tvec if tvec is None else tvec
    objects = np.column_stack((pnp.obj_pts, np.zeros(len(pnp.obj_pts))))
    pixels, _ = cv2.projectPoints(objects, rvec, tvec, camera.K, camera.D)
    rms = np.sqrt(np.mean(np.sum((pixels.reshape(-1, 2)-pnp.img_pts)**2, axis=1)))
    return {"rvec": np.asarray(rvec).ravel().tolist(), "tvec": np.asarray(tvec).ravel().tolist(),
            "reprojection_rms_px": float(rms)}


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1))))


def summarize(records):
    summary = {}
    for method in ("rgb", "subpixel", "depth"):
        steps, translations, errors, residuals, timings = [], [], [], [], []
        previous = None
        for frame in records:
            pose = frame.get(method)
            if pose is None:
                previous = None
                continue
            rotation = cv2.Rodrigues(np.array(pose["rvec"]))[0]
            translation = np.array(pose["tvec"])
            errors.append(pose["reprojection_rms_px"])
            if previous is not None:
                old_rotation, old_translation = previous
                steps.append(angle(rotation, old_rotation))
                translations.append(float(np.linalg.norm(translation-old_translation))*1000)
            previous = rotation, translation
            if "plane_rms_mm" in pose:
                residuals.append(pose["plane_rms_mm"])
                timings.append(pose["depth_ms"])
        def stats(values):
            return None if not values else {"median": float(np.median(values)),
                                           "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}
        summary[method] = {"frames": len(errors), "rotation_step_deg": stats(steps),
                           "translation_step_mm": stats(translations), "reprojection_px": stats(errors)}
        if residuals:
            summary[method].update(plane_rms_mm=stats(residuals), depth_ms=stats(timings))
    return summary


def common_steps(records):
    """Compare only identical adjacent frame pairs, never bridge detection gaps."""
    methods = ("rgb", "subpixel", "depth")
    values = {name: [] for name in methods}
    for a, b in zip(records, records[1:]):
        if not all(name in f for name in methods for f in (a, b)):
            continue
        for name in methods:
            values[name].append(angle(cv2.Rodrigues(np.array(b[name]["rvec"]))[0],
                                      cv2.Rodrigues(np.array(a[name]["rvec"]))[0]))
    return {name: {"pairs": len(steps), "median": float(np.median(steps)),
                   "p95": float(np.percentile(steps, 95)), "max": float(np.max(steps))}
            for name, steps in values.items() if steps}


def run(bag, *, start_frame=0, stop_frame=None, plane_scale=1.0, max_plane_rms=0.005):
    records = []
    with RealSenseReader(bag, rotations=1) as source:
        raw = make_estimator(source.calibration.color, tracking=False)
        # Separate config: do not mutate the shared BOARD detector.
        config = copy.copy(BOARD)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        config.detector = cv2.aruco.ArucoDetector(config.dictionary, parameters)
        refined = make_estimator(source.calibration.color, config, tracking=False)
        c = source.calibration
        projector = Projector(c.depth.K, c.color.K, c.depth_to_color,
                              (c.depth.height, c.depth.width), (c.color.height, c.color.width),
                              depth_D=c.depth.D, color_D=c.color.D,
                              depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)
        stop_frame = len(source) if stop_frame is None else min(stop_frame, len(source))
        if not 0 <= start_frame < stop_frame or plane_scale < 1 or max_plane_rms <= 0:
            raise ValueError("Require a nonempty frame range, plane-scale >= 1, and positive RMS limit")
        for index in range(start_frame, stop_frame):
            frame = source[index]
            image = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
            record = {"frame": index, "timestamp_ns": frame.timestamp_ns}
            for name, estimator in (("rgb", raw), ("subpixel", refined)):
                pose = estimator.get_board_transform(image)
                if pose is None:
                    continue
                pnp = pose[1]
                record[name] = pose_record(pnp, c.color)
                record[name]["markers"] = len(pnp.img_pts)//4
                if name == "subpixel":
                    try:
                        rvec, tvec, stats = depth_pose(frame, c.color, pnp, projector,
                                                      plane_scale=plane_scale, max_plane_rms=max_plane_rms)
                        record["depth"] = pose_record(pnp, c.color, rvec, tvec) | stats
                    except ValueError as exc:
                        record["depth_error"] = str(exc)
            records.append(record)
            if index % 30 == 0:
                print(f"Frame {index}/{stop_frame-1}", flush=True)
        report = {"bag": str(bag), "start_frame": start_frame, "stop_frame": stop_frame,
                  "fps": source.fps, "plane_scale": plane_scale, "max_plane_rms": max_plane_rms,
                  "summary": summarize(records), "common_rotation_steps_deg": common_steps(records), "frames": records}
    output = Path("exports") / (Path(bag).stem + f"_depth_pose_s{plane_scale:g}_r{max_plane_rms:g}.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print("Common frame-pair rotation changes:", json.dumps(report["common_rotation_steps_deg"], indent=2))
    print(f"Saved {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", default="recordings/20261005_194727.db3")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stop-frame", type=int, help="exclusive final frame")
    parser.add_argument("--plane-scale", type=float, default=1.0,
                        help="depth ROI expansion; >1 assumes surrounding surface is coplanar")
    parser.add_argument("--max-plane-rms", type=float, default=0.005, help="plane residual limit in meters")
    args = parser.parse_args()
    run(args.bag, start_frame=args.start_frame, stop_frame=args.stop_frame,
        plane_scale=args.plane_scale, max_plane_rms=args.max_plane_rms)
