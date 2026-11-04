"""Plane transport through board conventions, live validation and reacquisition."""

from dataclasses import dataclass
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from board_est import PnpResult
from rgbd_geometry import depth_points_to_reference, reference_points_to_depth
from rgbd_table_plane import BoardTabletopSegmenter, fit_plane, plane_points
from rgbd_tabletop_segmentation import segment_tabletop_objects, TabletopSegmentationParameters
from rgbd_types import Camera, Calibration


@dataclass
class Result:
    plane_model: tuple
    timings_ms: dict


class PlaneTests(unittest.TestCase):
    def setUp(self):
        camera = Camera(200, 160, np.array([[180., 0, 93], [0, 180, 65], [0, 0, 1]]), np.zeros(5), 'none')
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = cv2.Rodrigues(np.array([.01, -.02, .03]))[0]
        extrinsics[:3, 3] = [.014, -.006, .002]
        self.c = Calibration(camera, camera, extrinsics)
        pnp = PnpResult(np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(3), np.zeros(3))
        pnp.image_size = (200, 160)
        self.pose = (np.eye(4), pnp)
        self.plane = np.array([0., -1, 0, .2])
        self.parameters = TabletopSegmentationParameters(max_depth=1.1, plane_min_inliers=80)

    def depth_for(self, plane):
        y, x = np.indices((160, 200))
        rays = np.stack(((x-93)/180, (y-65)/180, np.ones(x.shape)), axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = -plane[3]/(rays @ plane[:3])
        depth[(depth < .25) | (depth > 1.1)] = 0
        return depth.astype(np.float32)

    def test_point_roundtrip_off_center_intrinsics_extrinsics_and_pixel_flip(self):
        self.c.color.D = np.array([.01, -.002, .001, -.001, 0])
        points = np.array([[.1, .2, .7], [-.2, .3, 1], [.3, -.1, 1.2]])
        reference = depth_points_to_reference(points, self.c, self.pose)
        returned = reference_points_to_depth(reference, self.c, self.pose)
        np.testing.assert_allclose(returned, points, atol=1e-7)

    def test_moving_camera_uses_board_plane_instead_of_reusing_camera_plane(self):
        tracker = BoardTabletopSegmenter(self.c, self.parameters)
        def segment(depth, camera, parameters, **kwargs):
            plane = kwargs['plane_model']
            if plane is None and tracker.reference_plane is not None:
                raise AssertionError('Supported plane should bypass global search')
            return Result(tuple(self.plane if plane is None else plane), {'total': 0})
        with patch('rgbd_table_plane.segment_tabletop_objects', side_effect=segment) as call:
            tracker.segment(self.depth_for(self.plane), pose=self.pose)
            stored = tracker.reference_plane.copy()
            reference = depth_points_to_reference(plane_points(self.plane, np.array([0, .2, .7])), self.c, self.pose)
            transform = np.eye(4)
            transform[:3, :3] = cv2.Rodrigues(np.array([.07, .03, -.02]))[0]
            transform[:3, 3] = [.04, .025, .02]
            moved = transform, self.pose[1]
            expected, _ = fit_plane(reference_points_to_depth(reference, self.c, moved))
            result = tracker.segment(self.depth_for(expected), pose=moved)
        self.assertEqual(tracker.status, 'tracked')
        self.assertGreater(tracker.support_count, 25)
        self.assertIsNotNone(call.call_args.kwargs['plane_model'])
        self.assertGreater(np.linalg.norm(expected-self.plane), .01)
        np.testing.assert_allclose(result.plane_model, expected, atol=1e-6)
        np.testing.assert_allclose(tracker.reference_plane, stored, atol=1e-6)

    def test_missing_pose_searches_fresh_and_depth_loss_does_not_invent_support(self):
        tracker = BoardTabletopSegmenter(self.c, self.parameters)
        depth = self.depth_for(self.plane)
        with patch('rgbd_table_plane.segment_tabletop_objects', return_value=Result(tuple(self.plane), {'total': 0})) as call:
            tracker.segment(depth, pose=self.pose)
            stored = tracker.reference_plane.copy()
            tracker.segment(depth, pose=None)
            self.assertIsNone(call.call_args.kwargs['plane_model'])
            np.testing.assert_array_equal(tracker.reference_plane, stored)
        with patch('rgbd_table_plane.segment_tabletop_objects', side_effect=RuntimeError('no plane')) as call:
            with self.assertRaises(RuntimeError):
                tracker.segment(np.zeros_like(depth), pose=self.pose)
            self.assertIsNone(call.call_args.kwargs['plane_model'])
        self.assertEqual(tracker.status, 'reacquiring')
        with patch('rgbd_table_plane.segment_tabletop_objects', return_value=Result(tuple(self.plane), {'total': 0})):
            tracker.segment(depth, pose=self.pose)
            self.assertEqual(tracker.status, 'tracked')
        tracker.reset()
        self.assertIsNone(tracker.reference_plane)

    def test_reacquired_surface_replaces_old_reference(self):
        tracker = BoardTabletopSegmenter(self.c, self.parameters)
        with patch('rgbd_table_plane.segment_tabletop_objects', return_value=Result(tuple(self.plane), {'total': 0})):
            tracker.segment(self.depth_for(self.plane), pose=self.pose)
        changed = np.array([0., -1, 0, .3])
        with patch('rgbd_table_plane.segment_tabletop_objects', return_value=Result(tuple(changed), {'total': 0})) as call:
            tracker.segment(self.depth_for(changed), pose=self.pose)
            self.assertIsNone(call.call_args.kwargs['plane_model'])
        actual, _ = fit_plane(reference_points_to_depth(plane_points(tracker.reference_plane, tracker.reference_center), self.c, self.pose))
        np.testing.assert_allclose(actual, changed, atol=1e-6)

    def test_half_resolution_known_plane_skips_ransac_and_restores_masks(self):
        result = segment_tabletop_objects(self.depth_for(self.plane), self.c.depth, self.parameters,
                                           processing_scale=.5, plane_model=self.plane)
        self.assertEqual(result.timings_ms['plane_ransac'], 0)
        self.assertEqual(result.table_mask.shape, (160, 200))
        self.assertGreater(result.table_mask.sum(), 100)
        np.testing.assert_allclose(result.plane_model, self.plane)


if __name__ == '__main__':
    unittest.main()
