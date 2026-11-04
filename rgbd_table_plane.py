"""A support plane anchored to the board, validated by fresh depth every frame."""

from dataclasses import replace
from time import perf_counter

import numpy as np

from rgbd_geometry import depth_points_to_reference, reference_points_to_depth
from rgbd_tabletop_segmentation import TabletopSegmentationParameters, segment_tabletop_objects


def fit_plane(points):
    center = points.mean(axis=0)
    values, vectors = np.linalg.eigh((points-center).T @ (points-center)/len(points))
    if values[1] < .02**2:
        raise ValueError('Plane support must span a surface, not a line')
    normal = vectors[:, 0]
    plane = np.r_[normal, -normal @ center]
    return (-plane if plane[3] < 0 else plane), center


def plane_points(plane, center):
    """Five well-spaced points on the plane, near its measured footprint."""
    normal = plane[:3]
    center = center-(center @ normal+plane[3])*normal
    first = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
    first *= .15/np.linalg.norm(first)
    second = np.cross(normal, first)
    return np.array([center, center+first, center-first, center+second, center-second])


class BoardTabletopSegmenter:
    """Depth + filtered RGB board pose -> objects on one remembered support.

    Store a plane and footprint center in reference coordinates, in meters.
    Current depth validates/refits the projection; unsupported priors trigger
    RANSAC. Conversions preserve the board estimator's pixel convention.
    """

    def __init__(self, calibration, parameters=None, *, processing_scale=.5):
        self.calibration = calibration
        self.parameters = parameters or TabletopSegmentationParameters()
        self.processing_scale = processing_scale
        self.reference_plane = self.reference_center = None
        self.status, self.support_count, self.residual_mm = 'searching', 0, None
        camera = calibration.depth
        stride = 8 if processing_scale == .5 else 4
        offset = 1 if processing_scale == .5 else 0
        self.y, self.x = np.mgrid[offset:camera.height:stride, offset:camera.width:stride]
        self.rays = np.stack(((self.x-camera.K[0, 2])/camera.K[0, 0],
                              (self.y-camera.K[1, 2])/camera.K[1, 1], np.ones(self.x.shape)), axis=-1)

    def reset(self):
        self.reference_plane = self.reference_center = None
        self.status, self.support_count, self.residual_mm = 'searching', 0, None

    def _samples(self, depth):
        p = self.parameters
        z = depth[self.y, self.x]
        valid = (np.isfinite(z) & (z >= p.min_depth) & (z <= p.max_depth)
                 & (self.y >= p.plane_roi_top*depth.shape[0]))
        return self.rays[valid]*z[valid, None]

    def _validate(self, plane, points):
        minimum = max(25, round(self.parameters.plane_min_inliers*self.processing_scale**2))
        members = points[np.abs(points @ plane[:3]+plane[3]) <= .015]
        self.support_count = len(members)
        if len(members) < minimum:
            return None
        try:
            for _ in range(2):
                fitted, center = fit_plane(members)
                if abs(fitted[:3] @ plane[:3]) < np.cos(np.deg2rad(5)):
                    return None
                keep = np.abs(members @ fitted[:3]+fitted[3]) <= self.parameters.plane_distance_threshold
                members = members[keep]
                if len(members) < minimum:
                    return None
            fitted, center = fit_plane(members)
        except ValueError:
            return None
        self.support_count = len(members)
        self.residual_mm = float(np.sqrt(np.mean((members @ fitted[:3]+fitted[3])**2))*1000)
        return fitted, center

    def _remember(self, plane, center, pose):
        reference = depth_points_to_reference(plane_points(plane, center), self.calibration, pose)
        measured, center = fit_plane(reference)
        if self.reference_plane is not None:
            if measured[:3] @ self.reference_plane[:3] < 0:
                measured = -measured
            measured = .95*self.reference_plane+.05*measured
            measured /= np.linalg.norm(measured[:3])
        self.reference_plane, self.reference_center = measured, center

    def segment(self, depth, *, pose=None):
        started = perf_counter()
        points = self._samples(depth)
        plane, self.support_count, self.residual_mm = None, 0, None
        if pose is not None and self.reference_plane is not None:
            reference = plane_points(self.reference_plane, self.reference_center)
            predicted, _ = fit_plane(reference_points_to_depth(reference, self.calibration, pose))
            supported = self._validate(predicted, points)
            if supported is not None:
                plane, center = supported
                self.status = 'tracked'
                self._remember(plane, center, pose)
            else:
                self.status = 'reacquiring'
        validation_ms = (perf_counter()-started)*1000
        result = segment_tabletop_objects(depth, self.calibration.depth, self.parameters,
            processing_scale=self.processing_scale, plane_model=plane)
        if plane is None:
            self.status = 'search (no board pose)' if pose is None else 'acquired'
            if pose is not None:
                members = points[np.abs(points @ result.plane_model[:3]+result.plane_model[3]) <= .015]
                self.support_count = len(members)
                if len(members) >= 3:
                    self.reference_plane = None  # Reacquisition replaces an unsupported prior.
                    self._remember(np.asarray(result.plane_model), members.mean(axis=0), pose)
        timings = dict(result.timings_ms, plane_tracking=validation_ms, total=(perf_counter()-started)*1000)
        return replace(result, timings_ms=timings)
