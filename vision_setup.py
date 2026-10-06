import os
import threading

import numpy as np
import cv2
import cam_config
import board_config
import board_est
import plotter2d
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class VisionSetup:
    camera: object
    board: board_config.BoardConfig
    detector: cv2.aruco.ArucoDetector
    marker_height_m: float
    plot_margin_m: float = 0.2
    rotate_180: bool = False
    board_estimator: board_est.BoardEstimator = field(init=False, repr=False)
    plotter: plotter2d.BoardPlotter2D = field(init=False, repr=False)

    def __post_init__(self):
        self.board_estimator = board_est.BoardEstimator(
            board_config=self.board,
            K=self.camera.K,
            D=self.camera.D,
            rotate_180=self.rotate_180,
        )
        self.plotter = plotter2d.BoardPlotter2D(
            self.board,
            self.detector.getDictionary(),
            margin=self.plot_margin_m,
        )

class _LatestWebcam:
    """Capture the next camera frame while the current one is processed."""

    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.delivered_sequence = 0
        self.closed = False
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        while not self.closed:
            success, frame = self.cap.read()
            if not success:
                break
            with self.condition:
                self.frame = frame
                self.sequence += 1
                self.condition.notify_all()
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def read(self):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != self.delivered_sequence or self.closed,
                timeout=1.0,
            )
            if self.sequence == self.delivered_sequence:
                return None
            self.delivered_sequence = self.sequence
            return self.frame

    def release(self):
        self.closed = True
        self.cap.release()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=1.0)


def _get_webcam_image():
    source = getattr(_get_webcam_image, "source", None)
    if source is None:
        source = _LatestWebcam()
        _get_webcam_image.source = source
        _get_webcam_image.cap = source.cap
    return source.read()

_webcam = cam_config.Camera(
    K=np.array([[735.09668766, 0., 308.18011975], [0., 735.62248422, 242.58646203], [0., 0., 1.]], dtype=np.float32),
    D=np.array([0.15017654, -1.34648531, 0.00405315, -0.00410719, 2.41472656], dtype=np.float32),
    frame_getter=_get_webcam_image,
)

_gridboard_letter = board_config.GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    size=(3, 4),
    marker_length=0.025,
    board_width=0.15,      # board content width (fits Letter height)
    print_width=0.2159,    # Letter width is 8.5" = 21.59cm
    filename="resources/gridboard_letter"
)

_gridboard_90 = board_config.GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    size=(3, 4),
    marker_length=0.1,
    board_width=0.84,
    print_width=0.9,
    filename="resources/gridboard_90"
)

_gridboard_200 = board_config.GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    size=(3, 4),
    marker_length=0.15,
    board_width=1.9,      # board content width
    print_width=2.0,      # print width
    filename="resources/gridboard_200"
)

_gridboard_240 = board_config.GridboardConfig(
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    size=(5, 7),
    marker_length=0.24,
    board_width=2.28,      # board content width
    print_width=2.4,      # print width
    filename="resources/gridboard_240"
)


_aruco_detector4 = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
_aruco_detector5 = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
_aruco_detector6 = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50))

_SMALL_SETUP = VisionSetup(
    camera=_webcam,
    board=_gridboard_letter,
    detector=_aruco_detector4,
    marker_height_m=0.12,
)

_MID_SETUP = VisionSetup(
    camera=_webcam,
    board=_gridboard_90,
    detector=_aruco_detector4,
    marker_height_m=0.12,
)

_LARGE_SETUP = VisionSetup(
    camera=_webcam,
    board=_gridboard_200,
    detector=_aruco_detector4,
    marker_height_m=0.12,
)

_EXTRA_LARGE_SETUP = VisionSetup(
    camera=_webcam,
    board=_gridboard_240,
    detector=_aruco_detector4,
    marker_height_m=0.12,
)

VISION = _MID_SETUP


if __name__ == "__main__":
    setup = _LARGE_SETUP
    if (input("Generate board? (y/n):")).lower() == "y":
        board = setup.board
        # Example usage: save board image and PDF
        print(board.get_print_dimensions())  # Print dimensions including margins
        img = board.generate_image(filepath=board.image_path, width_px=2160*4)
        board.generate_pdf(img, board.pdf_path)
    
    if (input("Generate individual markers? (y/n):")).lower() == "y":
        # Draw and save markers for IDs 0 through n-1 in the script's directory
        aruco_dict = setup.detector.getDictionary()
        script_dir = f"./resources/individual_markers/"
        for marker_id in range(16):
            # generateImageMarker creates the marker image
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 120)
            filename = os.path.join(script_dir, f"aruco_{marker_id:02d}.png")
            cv2.imwrite(filename, marker_img)
            print(f"Saved {filename}")
