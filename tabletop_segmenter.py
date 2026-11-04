"""Depth tabletop segmentation with optional reference-anchored plane memory.

Run this module for a DB3 demo. No RGB, pose estimator, source, object IDs, or
GUI is required by TabletopSegmenter.segment(). Only one support is tracked.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from _tabletop_geometry import TabletopSegmentationParameters as Parameters
from _tabletop_geometry import segment_tabletop_objects
from rgbd_types import Calibration, Detection, Pose


@dataclass
class PlaneParameters:
    sample_stride: int = 8
    support_distance: float = .015
    min_support: int = 25
    min_spread: float = .02
    max_angle_degrees: float = 5.
    max_rms: float = .004  # Meters; otherwise reacquire using the full plane search.
    update_gain: float = .05


class _PlaneMemory:
    """Internal support memory; all coordinate changes are rigid plane transforms."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.reset()

    def reset(self):
        self.plane = self.reference_frame = None

    def predict(self, depth_to_reference, reference_frame):
        if reference_frame != self.reference_frame:
            self.reset()
        if self.plane is None:
            return None
        plane = depth_to_reference.T @ self.plane
        plane /= np.linalg.norm(plane[:3])
        return -plane if plane[3] < 0 else plane

    def validate(self, plane, points, minimum, distance):
        p = self.parameters
        members = points[np.abs(points @ plane[:3]+plane[3]) <= p.support_distance]
        for _ in range(2):
            if len(members) < max(p.min_support, minimum):
                return None
            center = members.mean(axis=0)
            values, vectors = np.linalg.eigh((members-center).T @ (members-center)/len(members))
            if values[1] < p.min_spread**2:
                return None
            normal = vectors[:, 0]
            fitted = np.r_[normal, -normal @ center]
            if fitted[3] < 0:
                fitted = -fitted
            if abs(fitted[:3] @ plane[:3]) < np.cos(np.deg2rad(p.max_angle_degrees)):
                return None
            members = members[np.abs(members @ fitted[:3]+fitted[3]) <= distance]
        if len(members) < max(p.min_support, minimum):
            return None
        rms = float(np.sqrt(np.mean((members @ fitted[:3]+fitted[3])**2))*1000)
        if rms > p.max_rms*1000:
            return None
        return fitted, len(members), rms

    def remember(self, plane, depth_to_reference, reference_frame, *, replace=False):
        measured = np.linalg.inv(depth_to_reference).T @ plane
        measured /= np.linalg.norm(measured[:3])
        if self.plane is not None and not replace and reference_frame == self.reference_frame:
            if measured[:3] @ self.plane[:3] < 0:
                measured = -measured
            gain = self.parameters.update_gain
            measured = (1-gain)*self.plane + gain*measured
            measured /= np.linalg.norm(measured[:3])
        self.plane, self.reference_frame = measured, reference_frame


@dataclass
class Segmentation:
    detections: tuple[Detection, ...]
    object_labels: np.ndarray  # Native depth resolution; -1 background, 0..N-1 detections.
    table_mask: np.ndarray
    plane: np.ndarray | None  # Unit depth-camera plane ax+by+cz+d=0, normal toward camera.
    plane_status: str
    timings_ms: dict[str, float]
    rejections: dict[str, int] = field(default_factory=dict)  # Cluster/part rejection counts.
    noise_voxels: int = 0
    support_count: int = 0
    plane_rms_mm: float | None = None
    error: str = ""


class TabletopSegmenter:
    def __init__(self, calibration: Calibration, parameters=None, *, processing_scale=.5,
                 plane_parameters=None):
        self.calibration = calibration
        self.parameters = Parameters() if parameters is None else parameters
        self.processing_scale = processing_scale
        self.plane_parameters = PlaneParameters() if plane_parameters is None else plane_parameters
        self._memory = _PlaneMemory(self.plane_parameters)

    @property
    def reference_plane(self):
        return None if self._memory.plane is None else self._memory.plane.copy()

    def reset(self):
        self._memory.reset()

    def segment(self, depth: np.ndarray, pose: Pose | None = None) -> Segmentation:
        started = perf_counter()
        p, tracking, camera = self.parameters, self.plane_parameters, self.calibration.depth
        self._memory.parameters = tracking
        p.validate()
        if depth.shape != (camera.height, camera.width) or depth.dtype.kind != 'f':
            raise ValueError("Expected floating-point depth in meters at calibrated native dimensions")
        if np.any(camera.D):
            raise ValueError("Depth must be rectified before segmentation (zero distortion)")
        if self.processing_scale not in (.5, 1.):
            raise ValueError("processing_scale must be 0.5 or 1")
        if (not isinstance(tracking.sample_stride, int) or tracking.sample_stride < 1
                or tracking.support_distance <= 0 or tracking.min_support < 3 or tracking.min_spread <= 0
                or not 0 < tracking.max_angle_degrees <= 90 or tracking.max_rms <= 0
                or not 0 <= tracking.update_gain <= 1):
            raise ValueError("Invalid plane tracking parameters")
        stride = tracking.sample_stride
        y, x = np.mgrid[1:camera.height:stride, 1:camera.width:stride]
        z = depth[y, x]
        valid = np.isfinite(z) & (z >= p.min_depth) & (z <= p.max_depth) & (y >= p.plane_roi_top*camera.height)
        points = np.column_stack(((x[valid]-camera.K[0, 2])*z[valid]/camera.K[0, 0],
                                 (y[valid]-camera.K[1, 2])*z[valid]/camera.K[1, 1], z[valid]))
        plane, support, rms = None, 0, None
        depth_to_reference = None if pose is None else pose.camera_to_reference @ self.calibration.depth_to_color
        if pose is not None:
            predicted = self._memory.predict(depth_to_reference, pose.reference_frame)
            if predicted is not None:
                minimum = round(p.plane_min_inliers*self.processing_scale**2)
                supported = self._memory.validate(predicted, points, minimum, p.plane_distance_threshold)
                if supported is not None:
                    plane, support, rms = supported
        validation_ms = (perf_counter()-started)*1000
        try:
            result = segment_tabletop_objects(depth, camera, p, processing_scale=self.processing_scale, plane_model=plane)
        except (RuntimeError, ValueError) as exc:
            return Segmentation((), np.full(depth.shape, -1, np.int32), np.zeros(depth.shape, bool),
                                None, 'unavailable', dict(total=(perf_counter()-started)*1000), error=str(exc))
        status = 'tracked' if plane is not None else ('acquired' if pose is not None else 'search (no pose)')
        if plane is None:
            distances = points @ np.array(result.plane_model[:3])+result.plane_model[3]
            distances = distances[np.abs(distances) <= p.plane_distance_threshold]
            support = len(distances)
            rms = None if not support else float(np.sqrt(np.mean(distances**2))*1000)
        if pose is not None:
            self._memory.remember(np.array(result.plane_model), depth_to_reference,
                                  pose.reference_frame, replace=plane is None)
        conversion_started = perf_counter()
        detections = tuple(Detection(result.object_labels == obj.label, np.array(obj.centroid_xyz),
            np.array(obj.extent_xyz), (*map(int, obj.contour_depth_pixels[0]), *map(int, obj.contour_depth_pixels[2])),
            obj.footprint_diameter, obj.max_height_above_plane) for obj in result.objects)
        timings = dict(result.timings_ms, plane_validation=validation_ms,
                       shared_output=(perf_counter()-conversion_started)*1000, total=(perf_counter()-started)*1000)
        return Segmentation(detections, result.object_labels, result.table_mask, np.array(result.plane_model),
                            status, timings, result.rejections, result.noise_voxel_count, support, rms)


def main():
    import json
    from dataclasses import asdict
    import cv2
    from board_pose_estimator import BoardPoseEstimator
    from pose_filter import PoseFilter
    from rgbd_db3_reader import RealSenseReader
    from rgbd_demo import demo_parser, resize, run_demo

    parser = demo_parser(__doc__)
    parser.add_argument('--no-pose', action='store_true', help='stateless depth-only baseline')
    parser.add_argument('--processing-scale', type=float, choices=(.5, 1.), default=.5)
    parser.add_argument('--max-footprint', type=float, default=.2, help='meters; 0 disables')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    with RealSenseReader(args.bag, rotations=args.rotations) as source:
        estimator, smoother = BoardPoseEstimator(source.calibration.color), PoseFilter()
        segmenter = TabletopSegmenter(source.calibration, Parameters(max_object_footprint=args.max_footprint or None),
                                     processing_scale=args.processing_scale)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            config = dict(bag=str(args.bag.resolve()), rotations=args.rotations, no_pose=args.no_pose,
                          processing_scale=segmenter.processing_scale, parameters=asdict(segmenter.parameters),
                          plane_parameters=asdict(segmenter.plane_parameters), calibration=asdict(source.calibration))
            (args.output/'config.json').write_text(json.dumps(config, default=lambda value: value.tolist(), indent=2))
        previous_index = None
        last_result = None
        colors = np.array([[60, 220, 60], [220, 100, 50], [60, 180, 240], [200, 80, 200],
                           [220, 210, 50], [120, 80, 240]], np.uint8)

        def process(frame, index):
            nonlocal previous_index, last_result
            if previous_index is not None and abs(index-previous_index) != 1:
                segmenter.reset()
                smoother.reset()
            previous_index = index
            pose = None if args.no_pose else smoother.update(estimator.estimate(frame))
            result = segmenter.segment(frame.depth, pose)
            last_result = result
            depth = cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(frame.depth)/2, 0, 1)*255), cv2.COLORMAP_BONE)
            depth[~np.isfinite(frame.depth) | (frame.depth <= 0)] = 0
            depth[result.table_mask] = depth[result.table_mask]//2 + [30, 20, 0]
            rgb = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
            for i, detection in enumerate(result.detections):
                color = colors[i % len(colors)]
                depth[detection.mask] = depth[detection.mask]//2+color//2
                x0, y0, x1, y1 = detection.box
                cv2.rectangle(depth, (x0, y0), (x1, y1), color.tolist(), 2)
                cv2.putText(depth, str(i), (x0, max(12, y0-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color.tolist(), 1)
                # Project measured object pixels only; no dense RGB-D map is needed.
                y, x = np.nonzero(detection.mask)
                z, k = frame.depth[y, x], source.calibration.depth.K
                points = np.column_stack(((x-k[0, 2])*z/k[0, 0], (y-k[1, 2])*z/k[1, 1], z))
                transform = source.calibration.depth_to_color
                points = points @ transform[:3, :3].T + transform[:3, 3]
                points = points[points[:, 2] > 0]
                if len(points):
                    pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                              source.calibration.color.K, source.calibration.color.D)[0].reshape(-1, 2)
                    left, top = np.floor(pixels.min(axis=0)).astype(int)
                    right, bottom = np.ceil(pixels.max(axis=0)).astype(int)
                    cv2.rectangle(rgb, (left, top), (right, bottom), color.tolist(), 2)
                    cv2.putText(rgb, str(i), (left, top-4), cv2.FONT_HERSHEY_SIMPLEX, .6, color.tolist(), 2)
            depth, rgb = resize(depth, 600), resize(rgb, 600)
            image = np.hstack((depth, rgb))
            lines = [f'{len(result.detections)} objects | plane {result.plane_status} | {result.timings_ms["total"]:.1f} ms',
                     'Rejected: '+str(result.rejections), result.error]
            for row, line in enumerate(lines):
                cv2.putText(image, line, (8, 20+22*row), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, line, (8, 20+22*row), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
            return image, dict(status=result.plane_status, objects=len(result.detections),
                plane=None if result.plane is None else result.plane.tolist(), error=result.error,
                reference_plane=None if segmenter.reference_plane is None else segmenter.reference_plane.tolist(),
                rejections=result.rejections, timings_ms=result.timings_ms, noise_voxels=result.noise_voxels,
                support_count=result.support_count, plane_rms_mm=result.plane_rms_mm,
                detections=[dict(box=d.box, centroid=d.centroid.tolist(), extent=d.extent.tolist(),
                                 footprint=d.footprint, height=d.height, pixels=int(d.mask.sum())) for d in result.detections])

        run_demo(source, process, args, title='Depth segmentation | RGB boxes')
        if args.output and last_result is not None:
            np.savez_compressed(args.output/'last_masks.npz', object_labels=last_result.object_labels,
                                table_mask=last_result.table_mask)


if __name__ == '__main__':
    main()
