"""Actual ORB signatures, moved/occluded objects, ambiguity and merge rejection."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rgbd_appearance import describe, compare, Observation, RGBObjectAppearance, Similarity
from rgbd_types import Camera, Calibration, Frame
from rgbd_visual_memory import AppearanceMemory


def signature(seed, color=(240, 240, 240)):
    image = np.full((160, 100, 3), color, np.uint8)
    rng = np.random.default_rng(seed)
    for x, y in rng.integers([15, 20], [85, 140], (65, 2)):
        cv2.circle(image, (int(x), int(y)), int(rng.integers(1, 4)), (15, 15, 15), -1)
    cv2.putText(image, f'BOX {seed}', (12, 55), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
    mask = np.zeros((160, 100), np.uint8)
    mask[10:150, 8:92] = 1
    return describe(image, mask)


class VisualMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b = signature(2), signature(5)
        cls.blue = signature(3, (30, 90, 230))
        cls.red = signature(4, (230, 40, 30))

    def observation(self, value, extent=(.03, .055, .14)):
        return Observation(value, extent)

    def test_orb_distinguishes_different_print_on_similar_white_packages(self):
        same, other = compare(self.a, self.a), compare(self.a, self.b)
        self.assertTrue(same.reliable)
        self.assertGreater(same.score-other.score, .08)

    def test_occluded_object_recovers_its_id_after_45cm_move(self):
        memory = AppearanceMemory(probation_period=0)
        obs = self.observation(self.a)
        memory.update([[0, 0, 1]], 0, observations=[obs])
        memory.remember_label(0, dict(label='Melatonina', score=1., ocr_confidence=.95, margin=.5))
        memory.update([], 2, observations=[])
        tracks = memory.update([[.45, 0, 1]], 1, observations=[obs])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].id, 0)
        self.assertEqual(tracks[0].recoveries, 1)
        self.assertEqual(tracks[0].label, 'Melatonina')
        self.assertEqual(tracks[0].appearance_verified_at, 3)
        self.assertEqual(memory.events[0]['kind'], 'appearance_recovery')

    def test_swapped_positions_follow_appearance_instead_of_old_locations(self):
        memory = AppearanceMemory(probation_period=0)
        observations = [self.observation(self.blue), self.observation(self.red)]
        memory.update([[0, 0, 1], [.3, 0, 1]], 0, observations=observations)
        tracks = memory.update([[.3, 0, 1], [0, 0, 1]], .1, observations=observations)
        self.assertEqual([(t.id, t.detection_index) for t in tracks], [(0, 0), (1, 1)])
        self.assertEqual(memory.next_id, 2)

    def test_identical_packages_do_not_get_arbitrary_long_distance_recovery(self):
        memory = AppearanceMemory(probation_period=0)
        obs = self.observation(self.a)
        memory.update([[0, 0, 1], [.25, 0, 1]], 0, observations=[obs, obs])
        memory.update([], 1, observations=[])
        tracks = memory.update([[.6, 0, 1]], .1, observations=[obs])
        self.assertFalse(tracks[0].visible)
        self.assertFalse(tracks[1].visible)
        self.assertEqual(tracks[-1].id, 2)

    def test_different_textured_white_box_is_not_recovered_by_color_alone(self):
        memory = AppearanceMemory(probation_period=0)
        memory.update([[0, 0, 1]], 0, observations=[self.observation(self.a)])
        memory.update([], 1, observations=[])
        tracks = memory.update([[.4, 0, 1]], .1, observations=[self.observation(self.b)])
        self.assertEqual([(t.id, t.visible) for t in tracks], [(0, False), (1, True)])

    def test_weak_new_face_keeps_nearby_positional_continuity_without_training(self):
        memory = AppearanceMemory(probation_period=0)
        obs = self.observation(self.a)
        track = memory.update([[0, 0, 1]], 0, observations=[obs])[0]
        memory.update([], .033, observations=[])
        with patch('rgbd_visual_memory.compare', return_value=Similarity(.43, .54, False)):
            memory.update([[.01, 0, 1]], .033, observations=[self.observation(self.b)])
        self.assertTrue(track.visible)
        self.assertEqual(memory.next_id, 1)
        self.assertEqual(len(track.templates), 1)
        self.assertIs(track.templates[0], self.a)
        self.assertEqual(track.appearance_verified_at, -1.)

    def test_hand_box_merge_does_not_steal_or_update_the_old_template(self):
        memory = AppearanceMemory(probation_period=0)
        obs = self.observation(self.a)
        track = memory.update([[0, 0, 1]], 0, observations=[obs])[0]
        merged = self.observation(self.a, (.1, .18, .3))
        tracks = memory.update([[.03, 0, 1]], .2, observations=[merged])
        self.assertEqual(len(tracks), 1)
        self.assertFalse(track.visible)
        self.assertEqual(len(track.templates), 1)
        self.assertIs(track.templates[0], self.a)
        self.assertEqual(memory.next_id, 1)
        memory.update([[.35, 0, 1]], .2, observations=[obs])
        self.assertTrue(track.visible)
        self.assertEqual(track.id, 0)

    def test_tall_existing_package_is_not_blocked_by_a_short_neighbor(self):
        memory = AppearanceMemory(probation_period=0)
        observations = [self.observation(self.blue, (.02, .04, .18)),
                        self.observation(self.red, (.02, .04, .07))]
        positions = [[0, 0, 1], [.08, 0, 1]]
        memory.update(positions, 0, observations=observations)
        tracks = memory.update(positions, .1, observations=observations)
        self.assertTrue(all(t.visible for t in tracks))
        self.assertEqual(memory.next_id, 2)

    def test_new_larger_unrelated_object_nearby_is_not_suppressed_as_a_merge(self):
        memory = AppearanceMemory(probation_period=0)
        memory.update([[0, 0, 1]], 0, observations=[self.observation(self.blue, (.02, .04, .07))])
        tracks = memory.update([[.08, 0, 1]], .1, observations=[self.observation(self.red, (.03, .06, .18))])
        self.assertEqual(tracks[-1].id, 1)
        self.assertTrue(tracks[-1].visible)

    def test_conflicting_ocr_freezes_gallery_and_good_updates_keep_original_view(self):
        memory = AppearanceMemory(probation_period=0)
        obs = self.observation(self.a)
        track = memory.update([[0, 0, 1]], 0, observations=[obs])[0]
        memory.remember_label(0, None, conflict=True)
        memory.update([[0, 0, 1]], 2, observations=[obs])
        self.assertEqual(len(track.templates), 1)
        memory.remember_label(0, None, conflict=False)
        for _ in range(5):
            memory.update([[0, 0, 1]], 1.1, observations=[obs])
        self.assertEqual(len(track.templates), 3)
        self.assertIs(track.templates[0], self.a)

    def test_expiry_starts_new_id_without_inheriting_a_product_label(self):
        memory = AppearanceMemory(probation_period=0)
        obs = self.observation(self.a)
        memory.update([[0, 0, 1]], 0, observations=[obs])
        memory.remember_label(0, dict(label='Melatonina', score=1., ocr_confidence=.95, margin=.5))
        tracks = memory.update([[0, 0, 1]], 31, observations=[obs])
        self.assertEqual(tracks[0].id, 1)
        self.assertIsNone(tracks[0].label)

    def test_background_texture_is_not_used_as_an_object_descriptor(self):
        width, height = 200, 160
        K = np.array([[180., 0, 100], [0, 180, 80], [0, 0, 1]])
        camera = Camera(width, height, K, np.zeros(5), 'none')
        extractor = RGBObjectAppearance(Calibration(camera, camera, np.eye(4)))
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, (height, width, 3), np.uint8)
        rgb[40:130, 70:130] = (200, 40, 20)
        labels = np.full((height, width), -1, np.int32)
        labels[40:130, 70:130] = 0
        segmentation = SimpleNamespace(object_labels=labels, objects=[SimpleNamespace(label=0, extent_xyz=(.03, .055, .14))])
        observation = extractor.observe(Frame(0, rgb, np.ones((height, width), np.float32)), segmentation)[0]
        self.assertIsNotNone(observation.signature)
        self.assertIsNone(observation.signature.descriptors)


if __name__ == '__main__':
    unittest.main()
