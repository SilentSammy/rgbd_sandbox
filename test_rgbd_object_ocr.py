"""Delayed OCR association, bounded scheduling and auditable crop coordinates."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rgbd_label_matching import LabelMatcher
from rgbd_object_ocr import ObjectOCR, summarize
from rgbd_ocr import OCRResult, OCRText
from rgbd_types import Frame


class ObjectOCRTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reader = SimpleNamespace(read_regions=Mock(return_value=[
            OCRResult(0, 0, 1., [OCRText('Melatonina', .92, [[1., 2.]]*4)]),
            OCRResult(1, 90, 1., [OCRText('Colgate', .9, [[31., 6.]]*4)]),
        ]))
        with patch('rgbd_object_ocr.RegionOCR', return_value=self.reader):
            self.ocr = ObjectOCR(self.tmp.name)
            self.ocr.future.result(timeout=2)
        self.frame = Frame(123, np.zeros((40, 60, 3), np.uint8), np.ones((20, 30)))
        self.seeds = [dict(id=7, confirmed=True, anchor=np.zeros(3), box=[0, 0, 19, 19]),
                      dict(id=11, confirmed=True, anchor=np.ones(3), box=[30, 5, 59, 29])]
        self.tracks = [SimpleNamespace(id=s['id'], confirmed=True, position=s['anchor']) for s in self.seeds]
        self.assertEqual(self.ocr.poll(self.tracks)['kind'], 'ocr_ready')

    def tearDown(self):
        self.ocr.close()
        self.tmp.cleanup()

    def complete(self, tracks=None):
        self.ocr.future.result(timeout=2)
        return self.ocr.poll(self.tracks if tracks is None else tracks)

    def test_region_results_follow_capture_ids_even_when_track_order_changes(self):
        self.assertTrue(self.ocr.submit(self.frame, self.seeds, 1.))
        self.assertFalse(self.ocr.submit(self.frame, self.seeds, 2.))
        event = self.complete(self.tracks[::-1])
        self.assertEqual(event['accepted_ids'], [7, 11])
        self.assertEqual(self.ocr.cache[7]['label'], 'Melatonina')
        self.assertEqual(self.ocr.cache[11]['label'], 'Colgate')
        self.assertEqual(self.ocr.cache[7]['timestamp_ns'], 123)
        self.assertTrue(list(Path(self.tmp.name).glob('ocr_*/id_7.png')))
        self.assertEqual(self.reader.read_regions.call_args.args[1], [[0, 0, 24, 24], [26, 1, 60, 34]])

    def test_clear_discards_a_pending_result_from_the_previous_generation(self):
        self.ocr.submit(self.frame, self.seeds, 0.)
        self.ocr.clear()
        self.assertEqual(self.complete()['accepted_ids'], [])
        self.assertFalse(self.ocr.cache)
        self.assertTrue(self.ocr.submit(self.frame, self.seeds, 0.))

    def test_expired_id_and_large_position_jump_reject_delayed_results(self):
        self.ocr.submit(self.frame, self.seeds, 0.)
        moved = [SimpleNamespace(id=7, confirmed=True, position=np.array([.2, 0, 0]))]
        self.assertEqual(self.complete(moved)['accepted_ids'], [])
        self.assertFalse(self.ocr.cache)

    def test_verified_visual_recovery_allows_delayed_ocr_after_a_large_move(self):
        self.ocr.submit(self.frame, self.seeds[:1], 2.)
        moved = [SimpleNamespace(id=7, confirmed=True, position=np.array([.4, 0, 0]),
                                 appearance_verified_at=3., association='appearance')]
        self.assertEqual(self.complete(moved)['accepted_ids'], [7])
        self.assertEqual(self.ocr.cache[7]['label'], 'Melatonina')

    def test_refresh_uses_capture_elapsed_time_and_force_can_reread_a_paused_frame(self):
        self.ocr.submit(self.frame, self.seeds, 0.)
        self.complete()
        self.assertFalse(self.ocr.submit(self.frame, self.seeds, 4.9))
        self.assertTrue(self.ocr.submit(self.frame, self.seeds, 5.))
        self.complete()
        self.assertTrue(self.ocr.submit(self.frame, self.seeds, 5., force=True))

    def test_conflicting_labels_remain_ambiguous_and_duplicate_directions_merge(self):
        matcher = LabelMatcher({'Melatonina': [], 'Colgate': []})
        readings = [OCRResult(0, 0, 1., [OCRText('Melatonina', .8, []), OCRText('Colgate', .9, [])]),
                    OCRResult(0, 180, 1., [OCRText('MELATONINA', .95, [])])]
        item = summarize(readings, self.seeds[:1], matcher)[0]
        self.assertIsNone(item['label'])
        self.assertTrue(item['ambiguous'])
        self.assertEqual(len(item['raw']), 2)
        self.assertEqual(item['raw'][0]['confidence'], .95)

    def read_again(self, texts, elapsed=5.):
        self.frame.timestamp_ns += 100
        self.reader.read_regions.return_value = [OCRResult(0, 0, 1., texts)]
        self.assertTrue(self.ocr.submit(self.frame, self.seeds[:1], elapsed, force=True))
        self.complete()
        return self.ocr.cache[7]

    def test_unreadable_ocr_keeps_best_label_and_its_original_evidence_time(self):
        self.ocr.submit(self.frame, self.seeds, 0.)
        self.complete()
        item = self.read_again([])
        self.assertEqual(item['label'], 'Melatonina')
        self.assertIsNone(item['latest_label'])
        self.assertEqual(item['raw'], [])
        self.assertTrue(item['label_remembered'])
        self.assertEqual(item['best_match']['timestamp_ns'], 123)
        self.assertEqual(item['timestamp_ns'], 223)

    def test_weaker_conflicting_label_is_visible_without_overwriting_best(self):
        self.ocr.submit(self.frame, self.seeds, 0.)
        self.complete()
        item = self.read_again([OCRText('Colgatc', .99, [])])
        self.assertEqual(item['label'], 'Melatonina')
        self.assertEqual(item['latest_label'], 'Colgate')
        self.assertTrue(item['label_conflict'])
        self.assertTrue(item['label_remembered'])

    def test_stronger_evidence_replaces_best_and_equal_quality_keeps_first(self):
        self.ocr.submit(self.frame, self.seeds, 0.)
        self.complete()
        item = self.read_again([OCRText('Melatonina', .99, [])])
        self.assertEqual(item['best_match']['ocr_confidence'], .99)
        self.assertEqual(item['best_match']['timestamp_ns'], 223)
        self.assertFalse(item['label_remembered'])
        item = self.read_again([OCRText('Melatonina', .99, [])], 10.)
        self.assertEqual(item['best_match']['timestamp_ns'], 223)

    def test_ambiguous_reading_preserves_best_and_forgetting_discards_it(self):
        self.ocr.submit(self.frame, self.seeds, 0.)
        self.complete()
        item = self.read_again([OCRText('Melatonina', .99, []), OCRText('Colgate', .99, [])])
        self.assertEqual(item['label'], 'Melatonina')
        self.assertTrue(item['latest_ambiguous'])
        self.assertTrue(item['label_conflict'])
        self.ocr.poll([])
        self.assertFalse(self.ocr.cache)


if __name__ == '__main__':
    unittest.main()
