import unittest

from ocr_label_setup import LABELS
from rgbd_label_matching import LabelMatcher, normalize


class LabelMatchingTests(unittest.TestCase):
    def setUp(self):
        self.matcher = LabelMatcher(LABELS)

    def test_normalization_and_alias(self):
        self.assertEqual(normalize(" NESCAFÉ! "), "nescafe")
        self.assertEqual(self.matcher.match("nescafe").label, "NESCAFÉ")
        self.assertTrue(self.matcher.match("Melatonin").accepted)
        self.assertTrue(self.matcher.match("Colgate'").accepted)

    def test_clear_typo(self):
        match = self.matcher.match("IRUPROFENO")
        self.assertEqual(match.label, "Ibuprofeno")
        self.assertTrue(match.accepted)
        self.assertAlmostEqual(match.score, 0.9)

    def test_uncertain_and_unknown(self):
        for text in ("Molotonlua", "NESOARE", "purple cup", "400 mg"):
            self.assertFalse(self.matcher.match(text).accepted, text)
        self.assertIsNone(self.matcher.match("Q"))
        self.assertIsNone(self.matcher.match(""))

    def test_runner_up_margin(self):
        match = LabelMatcher({"druga": [], "drugb": []}).match("drugc")
        self.assertGreaterEqual(match.score, 0.75)
        self.assertEqual(match.margin, 0)
        self.assertFalse(match.accepted)


if __name__ == "__main__":
    unittest.main()
