"""Synthetic checks of plane fitting, pose recovery, and coordinate conventions."""

import unittest

import cv2
import numpy as np

from board_depth_pose_experiment import fit_plane, rigid_pose, depth_pose, common_steps
from board_est import PnpResult
from rgbd_db3_reader import Camera, Frame
from rgbd_projection import Projector


class DepthPoseTests(unittest.TestCase):
    def test_plane_rejects_foreground_outliers(self):
        rng = np.random.default_rng(42)
        xy = rng.uniform(-0.1, 0.1, (1000, 2))
        z = 0.8 + 0.2*xy[:, 0] - 0.1*xy[:, 1] + rng.normal(0, 0.0003, 1000)
        points = np.column_stack((xy, z))
        points[:100, 2] -= 0.1
        normal, offset, keep = fit_plane(points)
        expected = np.array([-0.2, 0.1, 1.0])
        expected /= np.linalg.norm(expected)
        self.assertGreater(normal @ expected, 0.99999)
        self.assertFalse(keep[:100].any())
        self.assertAlmostEqual(offset, 0.8/np.linalg.norm([-0.2, 0.1, 1]), places=3)

    def test_rigid_alignment_has_no_reflection(self):
        rng = np.random.default_rng(1)
        objects = np.column_stack((rng.uniform(-0.1, 0.1, (50, 2)), np.zeros(50)))
        rotation = cv2.Rodrigues(np.array([0.2, -0.1, 0.3]))[0]
        translation = np.array([0.1, 0.2, 0.8])
        rvec, tvec = rigid_pose(objects, objects @ rotation.T + translation)
        np.testing.assert_allclose(cv2.Rodrigues(rvec)[0], rotation, atol=1e-12)
        np.testing.assert_allclose(tvec, translation, atol=1e-12)

    def test_depth_pose_preserves_existing_reflected_pixel_convention(self):
        h, w = 200, 160
        K = np.array([[250., 0, 77.3], [0, 250., 98.6], [0, 0, 1.]])
        camera = Camera(w, h, K, np.zeros(5), "none")
        objects = np.array([[-.1, -.1, 0], [.1, -.1, 0], [.1, .1, 0], [-.1, .1, 0]])
        rvec, tvec = np.array([.2, -.1, .3]), np.array([.01, -.01, .8])
        pixels, _ = cv2.projectPoints(objects, rvec, tvec, K, camera.D)
        pnp = PnpResult(objects, pixels, tvec, rvec)
        pnp.image_size = (w, h)
        y, x = np.indices((h, w))
        virtual = np.stack((w-x, h-y, np.ones((h, w))), axis=-1) @ np.linalg.inv(K).T
        normal = cv2.Rodrigues(rvec)[0][:, 2]
        depth = ((normal @ tvec) / (virtual @ normal)).astype(np.float32)
        frame = Frame(0, np.zeros((h, w, 3), np.uint8), depth)
        projector = Projector(K, K, np.eye(4), (h, w), (h, w))
        actual_rvec, actual_tvec, stats = depth_pose(frame, camera, pnp, projector)
        np.testing.assert_allclose(cv2.Rodrigues(actual_rvec)[0], cv2.Rodrigues(rvec)[0], atol=1e-5)
        np.testing.assert_allclose(actual_tvec, tvec, atol=1e-5)
        self.assertLess(stats["plane_rms_mm"], 0.001)

    def test_insufficient_and_degenerate_depth(self):
        for points in (np.zeros((50, 3)), np.zeros((200, 3))):
            with self.assertRaises(ValueError):
                fit_plane(points)

    def test_comparison_does_not_bridge_missing_frames(self):
        pose = {name: {"rvec": [0., 0., 0.]} for name in ("rgb", "subpixel", "depth")}
        result = common_steps([pose, {}, pose, pose])
        self.assertTrue(all(value["pairs"] == 1 for value in result.values()))


if __name__ == "__main__":
    unittest.main()
