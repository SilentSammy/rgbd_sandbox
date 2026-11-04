"""Voxel membership, native-size restoration, and scaled pixel thresholds."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import open3d as o3d

from rgbd_types import Camera, Calibration, Frame
from rgbd_object_persistence import Track
from rgbd_projection import Projector
from object_persistence_demo import annotate
from rgbd_tabletop_segmentation import (
    _voxelize, _segment_half_resolution, segment_tabletop_objects,
    TabletopSegmentationParameters, TabletopSegmentationResult, DetectedObject,
)


class VoxelTests(unittest.TestCase):
    def test_membership_and_centroids_match_open3d_trace(self):
        points = np.random.default_rng(2).uniform(-.025, .025, (100, 3))
        centroids, owners = _voxelize(points, .01)
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        old, _, trace = cloud.voxel_down_sample_and_trace(
            .01, points.min(axis=0)-.005, points.max(axis=0)+.005, False)
        self.assertEqual(len(old.points), len(centroids))
        for expected, members in zip(np.asarray(old.points), trace):
            indices = np.asarray(members)
            self.assertTrue(np.all(owners[indices] == owners[indices[0]]))
            np.testing.assert_allclose(centroids[owners[indices[0]]], expected, atol=1e-12)

    def test_empty_candidates(self):
        centroids, owners = _voxelize(np.empty((0, 3)), .005)
        self.assertEqual(centroids.shape, (0, 3))
        self.assertEqual(owners.shape, (0,))


class HalfResolutionTests(unittest.TestCase):
    def setUp(self):
        self.params = TabletopSegmentationParameters()
        self.camera = Camera(20, 16, np.eye(3), np.zeros(5), "none")
        self.depth = np.ones((16, 20), np.float32)
        labels = np.full((8, 10), -1, np.int32)
        labels[2:5, 1:4] = 0
        obj = DetectedObject(0, 9, 5, (.1, .2, 1), (0, 0, .9), (.2, .3, 1.1),
                             (.2, .3, .2), .1, (2, 3), np.int32([[1, 2], [3, 2], [3, 4], [1, 4]]))
        self.result = TabletopSegmentationResult((0, -1, 0, .3), labels == -1, labels,
            (obj,), 80, 10, 10, 5, 0, {"total": 1}, 0)

    def test_restores_masks_centers_contours_and_exact_native_padding(self):
        self.depth[4, 2] = 0  # Invalid native samples must not inherit a coarse label.
        with patch("rgbd_tabletop_segmentation.segment_tabletop_objects", return_value=self.result) as call:
            result = _segment_half_resolution(self.depth, self.camera, self.params)
        reduced, camera, parameters = call.call_args.args
        self.assertEqual(reduced.shape, (8, 10))
        self.assertIs(camera, self.camera)
        self.assertEqual(parameters.min_object_pixels, 125)
        self.assertEqual(parameters.split_min_child_pixels, 125)
        self.assertEqual(parameters.plane_min_inliers, 250)
        self.assertEqual(parameters.bottom_padding_pixels, 0)
        self.assertEqual(result.object_labels.shape, self.depth.shape)
        self.assertEqual(result.table_mask.shape, self.depth.shape)
        self.assertEqual(result.object_labels[4, 2], -1)
        self.assertFalse(result.table_mask[4, 2])
        self.assertEqual(result.objects[0].pixel_count, 35)
        np.testing.assert_array_equal(result.objects[0].contour_depth_pixels,
                                      [[2, 4], [7, 4], [7, 14], [2, 14]])
        np.testing.assert_allclose(result.objects[0].centroid_depth_pixel, (4.5, 6.5))
        self.assertEqual(result.objects[0].centroid_xyz, self.result.objects[0].centroid_xyz)
        self.assertEqual(result.bottom_padding_pixels, 5)
        self.assertEqual(result.processing_scale, .5)
        self.assertIn("restore_native", result.timings_ms)

    def test_unsupported_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            segment_tabletop_objects(self.depth, self.camera, processing_scale=.25)

    def test_reduced_display_projection_accepts_native_masks_and_missing_tracks(self):
        with patch("rgbd_tabletop_segmentation.segment_tabletop_objects", return_value=self.result):
            segmentation = _segment_half_resolution(self.depth, self.camera, self.params)
        K = self.camera.K.copy()
        K[0] *= .5
        K[1] *= .5
        K[:2, 2] -= .25
        projector = Projector(K, K, np.eye(4), (8, 10), (8, 10))
        frame = Frame(0, np.zeros((16, 20, 3), np.uint8), self.depth)
        pnp = SimpleNamespace(to_image_points=lambda points: np.asarray(points).reshape(-1, 2))
        tracks = (Track(0, np.array([0., 0., 1.]), confirmed=True, detection_index=0),
                  Track(1, np.array([0., 0., 1.]), confirmed=True))
        result = SimpleNamespace(frame=frame, pose=(np.eye(4), pnp),
                                 segmentation=segmentation, tracks=tracks)
        calibration = Calibration(self.camera, self.camera, np.eye(4))
        color, depth = annotate(result, calibration, projector, max_dimension=10)
        self.assertEqual(color.shape, (8, 10, 3))
        self.assertEqual(depth.shape, (16, 20, 3))
        self.assertTrue(np.any(color))


if __name__ == "__main__":
    unittest.main()
