"""Three-frame filtering, per-ID tracking, occlusion, and navigation checks."""

import copy
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from board_est import PnpResult
from board_pose_setup import BOARD
from board_pose_tracking import PoseWindow, TrackedBoard, BoardPoseTracker


def camera_pose(x, degrees=0):
    camera_to_board = np.eye(4)
    camera_to_board[:3, :3] = cv2.Rodrigues(np.array([0., 0., np.deg2rad(degrees)]))[0]
    camera_to_board[:3, 3] = (x, 0, 1)
    return np.linalg.inv(camera_to_board)


class PoseWindowTests(unittest.TestCase):
    def test_uses_only_last_three_camera_positions(self):
        window = PoseWindow(3)
        for x in (10, 20, 30, 40):
            pose = window.update(camera_pose(x))
        self.assertEqual(len(window.poses), 3)
        np.testing.assert_allclose(np.linalg.inv(pose)[:3, 3], [30, 0, 1])

    def test_rotations_cross_pi_without_averaging_to_zero(self):
        window = PoseWindow(3)
        window.update(camera_pose(0, 179))
        pose = window.update(camera_pose(0, -179))
        rotation = np.linalg.inv(pose)[:3, :3]
        self.assertAlmostEqual(np.linalg.norm(cv2.Rodrigues(rotation)[0]), np.pi)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)

    def test_linear_motion_lags_one_frame_not_three(self):
        window = PoseWindow(3)
        for x in (0, 1, 2):
            pose = window.update(camera_pose(x))
        self.assertAlmostEqual(np.linalg.inv(pose)[0, 3], 1)


class MarkerTrackingTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.copy(BOARD)
        self.tracker = TrackedBoard(self.config)
        self.image = np.full((280, 320, 3), 255, np.uint8)
        self.corners = []
        for marker, x in ((0, 40), (1, 190)):
            tile = cv2.aruco.generateImageMarker(BOARD.dictionary, marker, 80)
            self.image[80:160, x:x+80] = tile[..., None]
            self.corners.append(np.float32([[[x, 80], [x+79, 80], [x+79, 159], [x, 159]]]))
        self.config.detect_corners = Mock(return_value=(self.corners, np.array([[0], [1]], np.int32)))

    def test_missing_marker_keeps_id_while_other_is_detected(self):
        self.tracker.detect_corners(self.image)
        moved = cv2.warpAffine(self.image, np.float32([[1, 0, 3], [0, 1, 2]]), (320, 280), borderValue=(255,)*3)
        self.config.detect_corners.return_value = ([self.corners[1]+[3, 2]], np.array([[1]], np.int32))
        corners, ids = self.tracker.detect_corners(moved)
        self.assertEqual(self.tracker.detected_ids, (1,))
        self.assertEqual(self.tracker.tracked_ids, (0,))
        self.assertEqual(ids.ravel().tolist(), [0, 1])
        np.testing.assert_allclose(corners[0], self.corners[0]+[3, 2], atol=0.5)

    def test_occluded_marker_is_not_inferred_from_visible_neighbor(self):
        self.tracker.detect_corners(self.image)
        image = self.image.copy()
        image[70:170, 30:130] = 255
        self.config.detect_corners.return_value = ([self.corners[1]], np.array([[1]], np.int32))
        _, ids = self.tracker.detect_corners(image)
        self.assertEqual(ids.ravel().tolist(), [1])
        self.assertEqual(self.tracker.tracked_ids, ())

    def test_fallback_expires_after_three_frames(self):
        self.tracker.detect_corners(self.image)
        self.config.detect_corners.return_value = ((), None)
        for _ in range(3):
            _, ids = self.tracker.detect_corners(self.image)
            self.assertEqual(ids.ravel().tolist(), [0, 1])
        _, ids = self.tracker.detect_corners(self.image)
        self.assertIsNone(ids)


class BoardPoseTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = BoardPoseTracker(BOARD, np.eye(3), np.zeros(5))
        self.image = np.zeros((100, 100, 3), np.uint8)
        obj = np.float32([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        pnp = PnpResult(obj, np.zeros((4, 2)), np.zeros(3), np.zeros(3))
        pnp.image_size = (100, 100)
        def estimate(frame):
            return camera_pose(1, 15), pnp
        self.tracker.estimator.get_board_transform = Mock(side_effect=estimate)

    def test_jumps_and_reversals_reset_history(self):
        for index in (10, 11, 12):
            self.tracker.get_board_transform(self.image, frame_index=index)
        self.assertEqual(len(self.tracker.filter.poses), 3)
        self.tracker.get_board_transform(self.image, frame_index=11)
        self.assertEqual(len(self.tracker.filter.poses), 1)
        self.tracker.get_board_transform(self.image, frame_index=10)
        self.assertEqual(len(self.tracker.filter.poses), 2)
        self.tracker.get_board_transform(self.image, frame_index=30)
        self.assertEqual(len(self.tracker.filter.poses), 1)

    def test_filtered_pnp_matches_returned_transform(self):
        transform, pnp = self.tracker.get_board_transform(self.image, frame_index=0)
        np.testing.assert_allclose(cv2.Rodrigues(pnp.rvec)[0] @ np.diag([1, -1, -1]),
                                   transform[:3, :3], atol=1e-6)
        np.testing.assert_allclose(pnp.tvec, transform[:3, 3])

    def test_lost_pose_clears_filter_without_reusing_stale_pose(self):
        self.tracker.get_board_transform(self.image, frame_index=0)
        self.tracker.estimator.get_board_transform.side_effect = lambda frame: None
        self.assertIsNone(self.tracker.get_board_transform(self.image, frame_index=1))
        self.assertEqual(len(self.tracker.filter.poses), 0)


if __name__ == "__main__":
    unittest.main()
