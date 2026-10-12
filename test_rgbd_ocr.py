"""Coordinate and crop tests without loading OCR models."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from rgbd_ocr import RegionOCR


class RegionOCRTests(unittest.TestCase):
    def setUp(self):
        self.ocr = RegionOCR.__new__(RegionOCR)
        self.ocr.engine = "rapidocr"
        polygon = [[10, 10], [20, 10], [20, 20], [10, 20]]
        self.ocr.reader = lambda image: ([
            [polygon, "label", 0.9], [polygon, "noise", 0.1]
        ], None)
        self.image = np.zeros((100, 100, 3), np.uint8)

    def test_rotated_polygons_return_to_original_image(self):
        readings = self.ocr.read_regions(self.image, [(10, 20, 30, 60)])
        expected = [(14.75, 24.75), (24.25, 24.75), (24.25, 54.25), (14.75, 54.25)]
        for result, point in zip(readings, expected):
            self.assertEqual(len(result.texts), 1)
            np.testing.assert_allclose(result.texts[0].polygon[0], point)

    def test_region_indices_and_empty_detection(self):
        self.ocr.reader = lambda image: (None, None)
        readings = self.ocr.read_regions(self.image, [(0, 0, 10, 10), (20, 20, 40, 40)], rotations=(0,))
        self.assertEqual([r.region for r in readings], [0, 1])
        self.assertTrue(all(not r.texts for r in readings))

    def test_invalid_crop(self):
        with self.assertRaises(ValueError):
            self.ocr.read_regions(self.image, [(-1, 0, 10, 10)])


class FastRegionOCRTests(unittest.TestCase):
    def setUp(self):
        self.ocr = RegionOCR.__new__(RegionOCR)
        self.ocr.engine, self.ocr.fast = "rapidocr", True
        boxes = np.float32([[[10, 10], [20, 10], [20, 20], [10, 20]]])
        self.line = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        self.ocr.reader = SimpleNamespace(
            text_det=Mock(return_value=(boxes, 0)),
            get_crop_img_list=Mock(return_value=[self.line]),
            text_rec=Mock(side_effect=lambda lines: ([("label", .9)]*len(lines), 0)),
        )
        self.image = np.zeros((100, 100, 3), np.uint8)

    def test_four_directions_share_two_detectors_and_one_recognition_batch(self):
        readings = self.ocr.read_regions(self.image, [(10, 20, 30, 60)])
        self.assertEqual([r.rotation for r in readings], [0, 90, 180, 270])
        self.assertEqual(self.ocr.reader.text_det.call_count, 2)
        self.ocr.reader.text_rec.assert_called_once()
        lines = self.ocr.reader.text_rec.call_args.args[0]
        self.assertEqual(len(lines), 4)
        np.testing.assert_array_equal(lines[1], np.rot90(lines[0], 2))
        for index, point in enumerate([(14.75, 24.75), (24.25, 24.75)]):
            np.testing.assert_allclose(readings[index].texts[0].polygon[0], point)
            np.testing.assert_allclose(readings[index].texts[0].polygon,
                                       readings[index+2].texts[0].polygon)

    def test_batch_spans_all_regions_and_confidence_filter_is_preserved(self):
        self.ocr.reader.text_rec.side_effect = lambda lines: ([("noise", .1)]*len(lines), 0)
        readings = self.ocr.read_regions(self.image, [(0, 0, 30, 40), (40, 50, 80, 90)])
        self.assertEqual([r.region for r in readings], [0]*4+[1]*4)
        self.assertEqual(self.ocr.reader.text_det.call_count, 4)
        self.ocr.reader.text_rec.assert_called_once()
        self.assertTrue(all(not r.texts for r in readings))

    def test_no_text_does_not_run_recognizer(self):
        self.ocr.reader.text_det.return_value = (None, 0)
        readings = self.ocr.read_regions(self.image, [(0, 0, 20, 30)])
        self.assertEqual(len(readings), 4)
        self.ocr.reader.text_rec.assert_not_called()

    def test_explicit_single_rotation_keeps_original_image_coordinates(self):
        readings = self.ocr.read_regions(self.image, [(10, 20, 30, 60)], rotations=(3,))
        self.assertEqual(readings[0].rotation, 270)
        np.testing.assert_allclose(readings[0].texts[0].polygon[0], [14.75, 54.25])

    def test_invalid_rotation(self):
        with self.assertRaises(ValueError):
            self.ocr.read_regions(self.image, [(0, 0, 20, 30)], rotations=(4,))


if __name__ == "__main__":
    unittest.main()
