"""One-shot capture cleanup and rectified RGB/depth geometry, without hardware."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import stretch4_tabletop_demo as demo
from rgbd_types import Camera


class SnapshotTests(unittest.TestCase):
    def test_display_crop_preserves_unprojected_geometry(self):
        camera = Camera(800, 1280, np.array([[500., 0, 400], [0, 500, 640], [0, 0, 1]]),
                        np.zeros(4), 'none')
        image = np.zeros((1280, 800, 3), np.uint8)
        depth = np.ones((1280, 800), np.float32)
        rgb, sparse, dense, cropped = demo.crop_snapshot(image, depth, depth, camera, (350, 480), 800)
        self.assertEqual(rgb.shape, (208, 800, 3))
        self.assertEqual(cropped.K[1, 2], 80)
        self.assertEqual(camera.K[1, 2], 640)  # Original calibration is unchanged.
        np.testing.assert_array_equal(sparse, dense)
        self.assertAlmostEqual((600-camera.K[1, 2])/camera.K[1, 1],
                               (40-cropped.K[1, 2])/cropped.K[1, 1])

    def test_out_of_bounds_crop_is_rejected(self):
        camera = Camera(8, 8, np.eye(3), np.zeros(4), 'none')
        with self.assertRaises(ValueError):
            demo.crop_snapshot(np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)),
                               np.ones((8, 8)), camera, (350, 480), 800)

    def test_rectification_preserves_depth_and_alignment(self):
        camera = Camera(80, 60, np.array([[50., 0, 40], [0, 50, 30], [0, 0, 1]]),
                        np.zeros(4), 'fisheye')
        depth = np.zeros((60, 80), np.float32)
        depth[20:40, 25:55] = 1.25
        image = np.repeat((depth > 0)[..., None], 3, axis=2).astype(np.uint8)*255
        rgb, measured, dense, rectified = demo.rectify_snapshot(image, depth, camera, depth)
        np.testing.assert_array_equal(measured, dense)
        self.assertTrue(np.all((dense == 0) | (dense == 1.25)))
        self.assertTrue(np.all(rgb[measured > 0] > 0))
        self.assertEqual(rectified.distortion_model, 'none')
        self.assertTrue(np.isfinite(rectified.K).all())

    def fake_api(self, usable=True):
        frame = SimpleNamespace(image=np.zeros((80, 60, 3), np.uint8),
                                depth_image=np.ones((80, 60), np.float32),
                                distortion_coefficients=[0., 0., 0., 0.])
        closed = Mock()
        def frames():
            try:
                yield None
                if usable:
                    for _ in range(5):
                        yield frame
            finally:
                closed()
        stream = Mock()
        stream.camera_matrices = {'right': np.array([[50., 0, 40], [0, 50, 30], [0, 0, 1]])}
        api = ModuleType('stretch4_emulated_rgbd.api')
        api.get_emulated_rgbd_stream = Mock(return_value=(stream, frames()))
        api.DenseDepthImage = Mock(return_value=SimpleNamespace(
            compute_dense_depth=lambda: frame.depth_image.copy()))
        return api, stream, closed

    def test_waits_for_valid_frame_and_releases_hardware(self):
        api, stream, closed = self.fake_api()
        with patch.dict(sys.modules, {'stretch4_emulated_rgbd.api': api}):
            image, measured, dense, camera = demo.capture_snapshot()
        stream.stop.assert_called_once()
        closed.assert_called_once()
        self.assertEqual(image.shape[:2], (80, 60))
        self.assertEqual((camera.width, camera.height), (60, 80))
        np.testing.assert_array_equal(measured, dense)
        self.assertTrue(api.get_emulated_rgbd_stream.call_args.kwargs['merge_lidars'])

    def test_releases_hardware_when_stream_has_no_frame(self):
        api, stream, closed = self.fake_api(usable=False)
        with patch.dict(sys.modules, {'stretch4_emulated_rgbd.api': api}):
            with self.assertRaisesRegex(RuntimeError, 'usable'):
                demo.capture_snapshot()
        stream.stop.assert_called_once()
        closed.assert_called_once()

    def test_waiting_in_gui_does_not_repeat_segmentation(self):
        result = SimpleNamespace(objects=(), table_mask=np.zeros((8, 8), bool),
                                 plane_model=(0, -1, 0, 0.5), timings_ms={})
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / 'input.npz'
            np.savez_compressed(snapshot, rgb=np.zeros((8, 8, 3), np.uint8),
                                measured_depth=np.ones((8, 8), np.float32),
                                depth=np.ones((8, 8), np.float32), K=np.eye(3))
            with patch.object(sys, 'argv', ['demo', '--input', str(snapshot),
                                           '--output-dir', directory]), \
                    patch.object(demo, 'segment_tabletop_objects', return_value=result) as segment, \
                    patch.object(cv2, 'imshow'), \
                    patch.object(cv2, 'waitKey', side_effect=[0, 0, ord('q')]), \
                    patch.object(cv2, 'getWindowProperty', return_value=1), \
                    patch.object(cv2, 'destroyAllWindows'):
                demo.main()
            segment.assert_called_once()


if __name__ == '__main__':
    unittest.main()
