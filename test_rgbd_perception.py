"""Stage contracts, reference geometry, capture timing, resets, and snapshots."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from board_est import PnpResult
from board_pose_setup import BOARD
from board_pose_tracking import BoardPoseEstimator, PoseFilter
from rgbd_geometry import image_point_to_reference, object_positions_in_reference, reference_points_to_image
from rgbd_object_persistence import ObjectMemory
from rgbd_perception import ObjectPersistor, PerceptionPipeline, TabletopSegmenter
from rgbd_types import Camera, Calibration, Frame


def pose(x=0):
    transform = np.eye(4)
    transform[:3, 3] = [x, 0, 1]
    pnp = PnpResult(np.zeros((8, 3)), np.zeros((8, 2)), np.zeros(3), np.zeros(3))
    pnp.image_size = (80, 60)
    return transform, pnp


class StageTests(unittest.TestCase):
    def test_rgb_estimator_does_not_filter_and_converts_rgb_at_its_boundary(self):
        estimator = BoardPoseEstimator(BOARD, np.eye(3), np.zeros(5))
        raw = pose(.5)
        estimator.estimator.get_board_transform = Mock(return_value=raw)
        rgb = np.full((60, 80, 3), [10, 20, 30], np.uint8)
        self.assertIs(estimator.estimate(rgb), raw)
        bgr = estimator.estimator.get_board_transform.call_args.args[0]
        np.testing.assert_array_equal(bgr[0, 0], [30, 20, 10])
        self.assertFalse(hasattr(estimator, "filter"))

    def test_filter_does_not_mutate_raw_pose_and_loss_resets_it(self):
        filter = PoseFilter(3)
        raw = pose(.5)
        for x in (0, .25):
            filter.update(pose(x))
        filtered = filter.update(raw)
        self.assertAlmostEqual(filtered[0][0, 3], .25)
        self.assertAlmostEqual(raw[0][0, 3], .5)
        np.testing.assert_array_equal(raw[1].tvec, np.zeros(3))
        np.testing.assert_allclose(filtered[1].tvec, filtered[0][:3, 3])
        self.assertIsNone(filter.update(None))
        self.assertEqual(len(filter.poses), 0)

    def test_segmenter_passes_depth_calibration_and_parameters(self):
        camera, parameters, depth = object(), object(), np.zeros((2, 2))
        with patch("rgbd_perception.segment_tabletop_objects", return_value="result") as segment:
            self.assertEqual(TabletopSegmenter(camera, parameters).segment(depth), "result")
            segment.assert_called_once_with(depth, camera, parameters, processing_scale=.5)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        K = np.array([[100., 0, 37], [0, 100, 28], [0, 0, 1]])
        camera = Camera(80, 60, K, np.zeros(5), "none")
        self.calibration = Calibration(camera, camera, np.eye(4))
        self.estimator = SimpleNamespace(estimate=Mock(return_value=pose()), reset=Mock(), status="RGB pose OK")
        self.detection = SimpleNamespace(centroid_xyz=(0., 0., 1.))
        self.segmentation = SimpleNamespace(objects=(self.detection,))
        self.segmenter = SimpleNamespace(segment=Mock(return_value=self.segmentation))
        self.memory = ObjectMemory(probation_period=.1, forget_after=2)
        self.pipeline = PerceptionPipeline(self.calibration, pose_estimator=self.estimator,
            segmenter=self.segmenter, persistor=ObjectPersistor(self.calibration, self.memory))

    def frame(self, seconds):
        return Frame(round(seconds*1e9), np.zeros((60, 80, 3), np.uint8), np.ones((60, 80), np.float32))

    def test_capture_dt_drives_probation_and_result_associates_current_detection(self):
        first = self.pipeline.process(self.frame(1))
        self.assertEqual(first.dt, 0)
        self.assertFalse(first.tracks[0].confirmed)
        second = self.pipeline.process(self.frame(1.1))
        self.assertAlmostEqual(second.dt, .1)
        self.assertTrue(second.tracks[0].confirmed)
        self.assertEqual(first.tracks[0].id, second.tracks[0].id)
        self.assertIs(second.segmentation.objects[second.tracks[0].detection_index], self.detection)
        self.assertEqual(set(second.timings_ms), {"pose", "filter", "segmentation", "persistence", "total"})

    def test_result_track_snapshots_do_not_change_with_later_frames(self):
        first = self.pipeline.process(self.frame(1))
        position = first.tracks[0].position.copy()
        self.detection.centroid_xyz = (.01, 0, 1)
        self.pipeline.process(self.frame(1.1))
        self.assertEqual(first.tracks[0].seen_seconds, 0)
        self.assertFalse(first.tracks[0].confirmed)
        np.testing.assert_array_equal(first.tracks[0].position, position)
        self.assertFalse(np.shares_memory(first.tracks[0].position, self.memory.tracks[0].position))

    def test_live_rejects_repeated_or_backward_time_before_processing(self):
        self.pipeline.process(self.frame(1))
        for seconds in (1, .9):
            with self.assertRaises(ValueError):
                self.pipeline.process(self.frame(seconds))
        self.assertEqual(self.estimator.estimate.call_count, 1)
        self.assertEqual(self.memory.elapsed_seconds, 0)

    def test_playback_reversals_and_jumps_reset_geometry_not_memory(self):
        self.pipeline.allow_reverse = True
        for index in (10, 11, 12):
            result = self.pipeline.process(self.frame(index/10), frame_index=index)
        self.assertEqual(len(self.pipeline.pose_filter.poses), 3)
        original_id = result.tracks[0].id
        result = self.pipeline.process(self.frame(1.1), frame_index=11)
        self.assertEqual(len(self.pipeline.pose_filter.poses), 1)
        self.assertEqual(result.tracks[0].id, original_id)
        self.assertAlmostEqual(result.dt, .1)
        self.pipeline.process(self.frame(1), frame_index=10)
        self.assertEqual(len(self.pipeline.pose_filter.poses), 2)
        self.pipeline.process(self.frame(1.5), frame_index=15)
        self.assertEqual(len(self.pipeline.pose_filter.poses), 1)
        self.assertEqual(self.estimator.reset.call_count, 2)

    def test_missing_pose_clears_filter_and_ages_confirmed_memory(self):
        self.pipeline.process(self.frame(1))
        accepted = self.pipeline.process(self.frame(1.1))
        self.estimator.estimate.return_value = None
        result = self.pipeline.process(self.frame(1.2))
        self.assertIsNone(result.pose)
        self.assertEqual(len(self.pipeline.pose_filter.poses), 0)
        self.assertFalse(result.tracks[0].visible)
        self.assertEqual(result.tracks[0].id, accepted.tracks[0].id)
        self.assertAlmostEqual(result.tracks[0].missing_seconds, .1)

    def test_support_plane_failure_is_reported_and_ages_memory(self):
        self.pipeline.process(self.frame(1))
        self.pipeline.process(self.frame(1.1))
        self.segmenter.segment.side_effect = RuntimeError("No support plane")
        result = self.pipeline.process(self.frame(1.2))
        self.assertIsNone(result.segmentation)
        self.assertEqual(result.status, "No support plane")
        self.assertFalse(result.tracks[0].visible)
        self.assertIsNotNone(result.pose)

    def test_new_session_resets_time_and_tracks_without_recycling_ids(self):
        self.pipeline.process(self.frame(1))
        self.pipeline.process(self.frame(1.1))
        self.pipeline.reset()
        result = self.pipeline.process(self.frame(0))
        self.assertEqual(result.dt, 0)
        self.assertEqual(result.tracks[0].id, 1)
        self.assertEqual(self.memory.elapsed_seconds, 0)
        self.assertEqual(len(self.pipeline.pose_filter.poses), 1)

    def test_image_dimensions_and_depth_units_are_checked(self):
        frame = self.frame(1)
        frame.timestamp_ns = float("nan")
        with self.assertRaises(ValueError):
            self.pipeline.process(frame)
        frame = self.frame(1)
        frame.depth = frame.depth.astype(np.uint16)
        with self.assertRaises(ValueError):
            self.pipeline.process(frame)
        frame = self.frame(1)
        frame.color = frame.color[:30]
        with self.assertRaises(ValueError):
            self.pipeline.process(frame)
        self.estimator.estimate.assert_not_called()

    def test_vectorized_reference_geometry_matches_original_pixel_mapping(self):
        transform = self.calibration.depth_to_color.copy()
        transform[:3, :3] = cv2.Rodrigues(np.array([.02, -.03, .04]))[0]
        transform[:3, 3] = [.01, -.02, .03]
        self.calibration.depth_to_color = transform
        self.calibration.color.D = np.array([.02, -.01, .001, -.002, 0])
        objects = [SimpleNamespace(centroid_xyz=p) for p in ((.1, .2, 1), (-.2, .1, 1.2))]
        raw_pose = pose(.05)
        points = np.array([o.centroid_xyz for o in objects]) @ transform[:3, :3].T + transform[:3, 3]
        camera = self.calibration.color
        pixels, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), camera.K, camera.D)
        expected = [image_point_to_reference(pixel, p[2], raw_pose, camera)
                    for pixel, p in zip(pixels.reshape(-1, 2), points)]
        positions = object_positions_in_reference(objects, self.calibration, raw_pose)
        np.testing.assert_allclose(positions, expected, atol=1e-12)
        projected, zs = reference_points_to_image(positions, raw_pose, camera)
        np.testing.assert_allclose(projected, pixels.reshape(-1, 2), atol=1e-6)
        np.testing.assert_allclose(zs, points[:, 2])


if __name__ == "__main__":
    unittest.main()
