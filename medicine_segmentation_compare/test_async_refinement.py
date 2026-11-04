"""Capture-time geometry and delayed-result safety, without model downloads."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from async_refinement import RefinedObject, mask_to_reference, usable_refinement
from live_refinement_demo import projected_refinement, coarse_seeds
from board_est import PnpResult
from rgbd_geometry import reference_points_to_image
from rgbd_types import Camera
from rgbd_projection import Projector


class DelayedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(80, 60, np.array([[100., 0, 38], [0, 100, 27], [0, 0, 1]]), np.zeros(5), 'none')
        pnp = PnpResult(np.zeros((4, 3)), np.zeros((4, 2)), np.zeros(3), np.zeros(3))
        pnp.image_size = (80, 60)
        self.pose = (np.eye(4), pnp)
        self.mask = np.zeros((60, 80), bool)
        self.mask[20:30, 25:35] = True
        y, x = np.indices((60, 80))
        self.inverse = np.stack((x, y), axis=-1)
        self.z = np.ones((60, 80), np.float32)
        self.seed = dict(z=1., z_band=.04, box=[24, 19, 35, 30])

    def test_capture_points_reproject_under_camera_motion_with_board_pixel_flip(self):
        points = mask_to_reference(self.mask, self.inverse, self.z, self.pose, self.camera, self.seed)
        pixels, z = reference_points_to_image(points, self.pose, self.camera)
        y, x = np.where(self.mask)
        np.testing.assert_allclose(pixels, np.column_stack((x, y)), atol=1e-5)
        moved = self.pose[0].copy()
        moved[0, 3] = .02
        projected, _ = reference_points_to_image(points, (moved, self.pose[1]), self.camera)
        np.testing.assert_allclose(projected, pixels-[2, 0], atol=1e-5)
        np.testing.assert_allclose(z, 1)

    def test_background_depth_and_missing_measurements_do_not_become_object_surface(self):
        self.z[20:25, 25:35] = 2.
        self.inverse[25:27, 25:35] = -1
        points = mask_to_reference(self.mask, self.inverse, self.z, self.pose, self.camera, self.seed)
        self.assertEqual(len(points), 30)
        np.testing.assert_allclose(points[:, 2], 1)
        self.z[:] = 2
        self.assertEqual(len(mask_to_reference(self.mask, self.inverse, self.z, self.pose, self.camera, self.seed)), 0)

    def test_old_results_reject_wrong_id_motion_loss_and_expiry(self):
        refined = RefinedObject(7, 1_000_000_000, np.zeros(3), np.zeros((20, 3)), .9)
        track = SimpleNamespace(id=7, visible=True, confirmed=True, position=np.zeros(3))
        self.assertTrue(usable_refinement(refined, track, 3_000_000_000))
        for changes in (dict(id=8), dict(visible=False), dict(confirmed=False), dict(position=np.array([.2, 0, 0]))):
            altered = SimpleNamespace(**(vars(track) | changes))
            self.assertFalse(usable_refinement(refined, altered, 3_000_000_000))
        self.assertFalse(usable_refinement(refined, track, 20_000_000_000))
        self.assertFalse(usable_refinement(refined, track, 0))

    def test_no_projection_without_pose_or_with_current_depth_mismatch(self):
        points = mask_to_reference(self.mask, self.inverse, self.z, self.pose, self.camera, self.seed)
        refined = RefinedObject(7, 1_000_000_000, np.zeros(3), points, .9)
        track = SimpleNamespace(id=7, visible=True, confirmed=True, position=np.zeros(3))
        result = SimpleNamespace(pose=self.pose, frame=SimpleNamespace(timestamp_ns=3_000_000_000))
        bounds, reason = projected_refinement(refined, track, self.seed, result, self.camera, 15.)
        self.assertEqual(reason, 'ok')
        np.testing.assert_allclose(bounds, [25, 20, 34, 29], atol=1)
        result.pose = None
        self.assertIsNone(projected_refinement(refined, track, self.seed, result, self.camera, 15.)[0])
        result.pose = self.pose
        self.assertEqual(projected_refinement(refined, track, self.seed | dict(z=1.3), result, self.camera, 15.)[1],
                         'current depth mismatch')

    def test_coarse_boxes_survive_missing_pose_and_no_tracks(self):
        labels = np.where(self.mask, 0, -1).astype(np.int32)
        result = SimpleNamespace(pose=None, tracks=(),
            frame=SimpleNamespace(depth=self.z, color=np.zeros((60, 80, 3), np.uint8)),
            segmentation=SimpleNamespace(object_labels=labels, objects=[SimpleNamespace(label=0)]))
        projector = Projector(self.camera.K, self.camera.K, np.eye(4), (60, 80), (60, 80))
        seeds = coarse_seeds(result, projector)
        self.assertEqual(len(seeds), 1)
        self.assertIsNone(seeds[0]['id'])
        self.assertFalse(seeds[0]['confirmed'])
        self.assertEqual(seeds[0]['label'], 0)
        self.assertLess(seeds[0]['box'][0], 25)


if __name__ == '__main__':
    unittest.main()
