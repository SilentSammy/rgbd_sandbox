"""Footprint gates reject wide background patches while retaining tall boxes."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from rgbd_types import Camera
from rgbd_tabletop_segmentation import footprint_diameter, segment_tabletop_objects, TabletopSegmentationParameters


class FootprintTests(unittest.TestCase):
    def test_height_is_not_counted_and_camera_rotation_preserves_diameter(self):
        points = np.array([[x, y, z] for x in (-.025, .025) for y in (0, .3) for z in (-.015, .015)])
        normal = np.array([0., -1, 0])
        expected = np.hypot(.05, .03)
        self.assertAlmostEqual(footprint_diameter(points, normal), expected, places=6)
        rotation = cv2.Rodrigues(np.array([.7, -.3, .4]))[0]
        self.assertAlmostEqual(footprint_diameter(points @ rotation.T, rotation @ normal), expected, places=6)

    def test_wide_patch_is_rejected_but_tall_narrow_box_remains(self):
        camera = Camera(320, 240, np.array([[300., 0, 160], [0, 300, 80], [0, 0, 1]]), np.zeros(5), 'none')
        y, x = np.indices((240, 320))
        depth = np.zeros((240, 320), np.float32)
        below = y > 80
        depth[below] = .2*300/(y[below]-80)
        depth[(depth < .25) | (depth > 1.1)] = 0
        depth[120:165, 90:190] = .7  # 23 cm wide wall-like patch.
        depth[90:165, 235:262] = .7  # Narrow box, taller than the footprint limit.
        params = TabletopSegmentationParameters(object_roi_top=.3, max_depth=1.1)
        before = segment_tabletop_objects(depth, camera, params, processing_scale=.5, plane_model=[0, -1, 0, .2])
        after = segment_tabletop_objects(depth, camera, replace(params, max_object_footprint=.15),
                                          processing_scale=.5, plane_model=[0, -1, 0, .2])
        self.assertEqual(len(before.objects), 2)
        self.assertEqual(len(after.objects), 1)
        self.assertGreater(max(before.objects, key=lambda o:o.footprint_diameter).footprint_diameter, .2)
        self.assertLess(after.objects[0].footprint_diameter, .1)
        self.assertGreater(after.objects[0].max_height_above_plane, .15)

    def test_invalid_limit(self):
        for value in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                TabletopSegmentationParameters(max_object_footprint=value).validate()
        TabletopSegmentationParameters(max_object_footprint=None).validate()


if __name__ == '__main__':
    unittest.main()
