"""Shared playback tests; no GUI windows or recordings needed."""

import unittest
from unittest.mock import patch

from rgbd_db3_viewer import Playback


class Recording:
    fps = 10

    def __len__(self):
        return 6


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.playback = Playback(Recording(), start_frame=2)

    def press(self, key, now=0):
        return self.playback.update(ord(key), now=now)

    def test_starts_paused(self):
        self.playback.update(now=100)
        self.assertEqual((self.playback.index, self.playback.direction), (2, 0))

    def test_play_reverse_and_pause(self):
        self.press("D")
        self.playback.update(now=0.05)
        self.assertEqual(self.playback.index, 2)
        self.playback.update(now=0.11)
        self.assertEqual(self.playback.index, 3)
        self.press("A", now=1)
        self.playback.update(now=1.11)
        self.assertEqual(self.playback.index, 2)
        self.press("S", now=2)
        self.playback.update(now=100)
        self.assertEqual((self.playback.index, self.playback.direction), (2, 0))

    def test_step_and_restart_pause(self):
        self.press("d")
        self.press("q", now=10)
        self.assertEqual((self.playback.index, self.playback.direction), (1, 0))
        self.press("e")
        self.press("e")
        self.assertEqual(self.playback.index, 3)
        self.press("a")
        self.press("w", now=10)
        self.assertEqual((self.playback.index, self.playback.direction), (2, 0))

    def test_boundaries_stop_and_steps_clamp(self):
        self.playback.seek(4)
        self.press("d")
        self.playback.update(now=1)
        self.assertEqual((self.playback.index, self.playback.direction), (5, 0))
        self.press("e")
        self.assertEqual(self.playback.index, 5)
        self.playback.seek(1)
        self.press("a")
        self.playback.update(now=1)
        self.assertEqual((self.playback.index, self.playback.direction), (0, 0))
        self.press("q")
        self.assertEqual(self.playback.index, 0)

    def test_slow_processing_does_not_skip_frames(self):
        self.press("d")
        self.playback.update(now=100)
        self.assertEqual(self.playback.index, 3)
        self.playback.update(now=100.01)
        self.assertEqual(self.playback.index, 3)

    def test_held_play_key_does_not_reset_timer(self):
        self.press("d")
        self.press("d", now=0.05)
        self.press("d", now=0.11)
        self.assertEqual(self.playback.index, 3)

    def test_quit_and_extra_commands(self):
        self.assertEqual(self.press("H"), "h")
        self.assertFalse(self.playback.quit)
        self.press("d")
        self.press("\x1b", now=100)
        self.assertTrue(self.playback.quit)
        self.assertEqual(self.playback.index, 2)

    def test_poll_processes_no_key_and_returns_extra_keys(self):
        self.press("d")
        with patch("rgbd_db3_viewer.cv2.waitKey", return_value=-1) as wait, \
                patch("rgbd_db3_viewer.perf_counter", return_value=1):
            self.assertEqual(self.playback.poll(), "")
            self.assertEqual(self.playback.index, 3)
            wait.assert_called_once_with(20)
        with patch("rgbd_db3_viewer.cv2.waitKey", return_value=ord("C")):
            self.assertEqual(self.playback.poll(), "c")

    def test_missing_fps_uses_default(self):
        source = Recording()
        source.fps = None
        self.assertAlmostEqual(Playback(source).interval, 1/30)

    def test_invalid_start_frame(self):
        for index in (-1, 6):
            with self.assertRaises(ValueError):
                Playback(Recording(), index)


if __name__ == "__main__":
    unittest.main()
