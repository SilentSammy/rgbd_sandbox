"""Shared letter GridBoard and RGB camera pose-estimator setup."""

import cv2
import numpy as np

from board_config import GridboardConfig
from board_est import BoardEstimator
from board_pose_tracking import BoardPoseTracker


DEFAULT_DICTIONARY = "4x4"
DEFAULT_BOARD_SCALE = 0.96
DEFAULT_ROTATIONS = 1  # Clockwise image/calibration quarter-turns in the reader.


def letter_board(dictionary=DEFAULT_DICTIONARY, board_scale=DEFAULT_BOARD_SCALE):
    """Letter board geometry with uniform printer scaling applied."""
    if not np.isfinite(board_scale) or board_scale <= 0:
        raise ValueError("board_scale must be finite and positive")
    dictionary_id = {
        "4x4": cv2.aruco.DICT_4X4_50,
        "5x5": cv2.aruco.DICT_5X5_50,
    }[dictionary]
    return GridboardConfig(
        dictionary=cv2.aruco.getPredefinedDictionary(dictionary_id),
        size=(3, 4),
        marker_length=0.025 * board_scale,
        board_width=0.15 * board_scale,
        print_width=0.2159 * board_scale,
        filename="resources/gridboard_letter",
    )


_LETTER_BOARD = letter_board()
BOARD = _LETTER_BOARD


def make_estimator(color_camera, board=None, *, tracking=True):
    """RGB-only tracked/three-frame-filtered pose; tracking=False is the raw baseline."""
    estimator = BoardPoseTracker if tracking else BoardEstimator
    return estimator(BOARD if board is None else board, color_camera.K, color_camera.D)
