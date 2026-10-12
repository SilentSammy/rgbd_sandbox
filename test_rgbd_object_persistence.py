"""Object memory tests without bags, segmentation, OCR, or GUI windows."""

import unittest

import numpy as np

from rgbd_object_persistence import ObjectMemory


class ObjectMemoryTests(unittest.TestCase):
    def test_probation_uses_only_dt(self):
        memory = ObjectMemory(probation_period=1)
        track = memory.update([[0, 0, 0]], 0)[0]
        memory.update([[0.01, 0, 0]], 0.5)
        self.assertFalse(track.confirmed)
        for _ in range(5):
            memory.update([[0.01, 0, 0]], 0)
        self.assertEqual(track.seen_seconds, 0.5)
        memory.update([[0, 0, 0]], 0.5)
        self.assertTrue(track.confirmed)
        self.assertEqual(track.id, 0)
        self.assertEqual(memory.elapsed_seconds, 1)

    def test_missing_probationary_track_is_discarded(self):
        memory = ObjectMemory()
        memory.update([[0, 0, 0]], 0)
        memory.update([], 0.2)
        self.assertEqual(memory.tracks, [])
        track = memory.update([[0, 0, 0]], 0.2)[0]
        self.assertEqual(track.id, 1)
        self.assertEqual(track.seen_seconds, 0)

    def test_confirmed_track_survives_gap_and_expires(self):
        memory = ObjectMemory(probation_period=0, forget_after=2)
        track = memory.update([[0, 0, 0]], 0)[0]
        memory.update([], 0.5)
        self.assertFalse(track.visible)
        self.assertEqual(track.missing_seconds, 0.5)
        memory.update([[0, 0, 0]], 0.5)
        self.assertIs(memory.tracks[0], track)
        self.assertTrue(track.visible)
        self.assertEqual(track.missing_seconds, 0)
        memory.update([], 2)
        self.assertEqual(memory.tracks, [])

    def test_large_jump_expires_before_matching(self):
        memory = ObjectMemory(probation_period=0, forget_after=2)
        memory.update([[0, 0, 0]], 0)
        tracks = memory.update([[0, 0, 0]], 3)
        self.assertEqual([t.id for t in tracks], [1])

    def test_reordering_detections_keeps_ids(self):
        memory = ObjectMemory(probation_period=0)
        memory.update([[0, 0, 0], [0.2, 0, 0]], 0)
        tracks = memory.update([[0.21, 0, 0], [0.01, 0, 0]], 0.1)
        self.assertEqual([(t.id, t.detection_index) for t in tracks], [(0, 1), (1, 0)])

    def test_gating_preserves_valid_one_to_one_matches(self):
        memory = ObjectMemory(probation_period=0, match_distance=0.11)
        memory.update([[0, 0, 0], [0.1, 0, 0]], 0)
        # Ungated minimum distance would choose a diagonal with an invalid pair.
        tracks = memory.update([[0, 0, 0], [0, 0.1, 0]], 0.1)
        self.assertEqual([(t.id, t.detection_index) for t in tracks], [(0, 1), (1, 0)])
        self.assertEqual(memory.next_id, 2)

    def test_far_detection_is_new_and_old_track_is_hidden(self):
        memory = ObjectMemory(probation_period=0)
        memory.update([[0, 0, 0]], 0)
        tracks = memory.update([[1, 0, 0]], 0.1)
        self.assertEqual([(t.id, t.visible) for t in tracks], [(0, False), (1, True)])

    def test_reverse_visits_accumulate_positive_video_time(self):
        memory = ObjectMemory(probation_period=0.2)
        previous = None
        for timestamp in (1.0, 1.1, 1.0):
            dt = 0 if previous is None else abs(timestamp - previous)
            track = memory.update([[0, 0, 0]], dt)[0]
            previous = timestamp
        self.assertTrue(track.confirmed)
        self.assertAlmostEqual(memory.elapsed_seconds, 0.2)
        self.assertEqual(track.id, 0)

    def test_clear_does_not_reuse_ids(self):
        memory = ObjectMemory()
        memory.update([[0, 0, 0]], 0)
        memory.clear()
        track = memory.update([[0, 0, 0]], 0)[0]
        self.assertEqual(track.id, 1)

    def test_invalid_input(self):
        for kwargs in ({"match_distance": 0}, {"probation_period": -1}, {"forget_after": 0}):
            with self.assertRaises(ValueError):
                ObjectMemory(**kwargs)
        memory = ObjectMemory()
        for positions, dt in (([], -1), ([], np.nan), ([[np.nan, 0, 0]], 0)):
            with self.assertRaises(ValueError):
                memory.update(positions, dt)


if __name__ == "__main__":
    unittest.main()
