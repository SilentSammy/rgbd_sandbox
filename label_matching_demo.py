"""Browse segmentation, raw OCR, and known-label suggestions with viewer playback.

python label_matching_demo.py --start-frame 137
Edit the vocabulary and acceptance thresholds in ocr_label_setup.py.
"""

import argparse

from ocr_label_setup import LABELS, MIN_MATCH_MARGIN, MIN_MATCH_SCORE
from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import PLAYBACK_HELP
from rgbd_label_matching import LabelMatcher
from segmentation_ocr_demo import view


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=PLAYBACK_HELP)
    parser.add_argument("bag", nargs="?", default="recordings/20261005_195417.db3")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-dimension", type=int, default=960)
    args = parser.parse_args()
    matcher = LabelMatcher(LABELS, MIN_MATCH_SCORE, MIN_MATCH_MARGIN)
    with RealSenseReader(args.bag, rotations=1) as reader:
        view(reader, start_frame=args.start_frame, max_dimension=args.max_dimension, label_matcher=matcher)
