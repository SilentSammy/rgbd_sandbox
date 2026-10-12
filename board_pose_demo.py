"""Step through a recording and visualize the RGB camera in the letter board frame.

Run: python board_pose_demo.py recordings/20261005_194250.db3
Focus the RGB window for the shared viewer playback controls (no Enter).
The board origin is its center; axes follow the existing BoardEstimator.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from board_est import BoardPlotter3D, get_cam_T
from board_pose_setup import (
    DEFAULT_BOARD_SCALE, DEFAULT_DICTIONARY, DEFAULT_ROTATIONS, letter_board, make_estimator,
)
from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _resize


def estimate_frame(source, index, estimator):
    """Return an annotated BGR image and the existing estimator's result."""
    frame = source[index]
    image = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
    result = estimator.get_board_transform(image, frame_index=index)
    lines = [f"Frame {index}/{len(source) - 1}", estimator.status]
    if result is None:
        lines.append("No board pose (requires at least 2 board markers)")
    else:
        board_to_camera, pnp = result
        camera_to_board = get_cam_T(board_to_camera)
        position = camera_to_board[:3, 3]
        object_points = np.column_stack((pnp.obj_pts, np.zeros(len(pnp.obj_pts))))
        projected, _ = cv2.projectPoints(
            object_points, pnp.rvec, pnp.tvec, estimator.K, estimator.D
        )
        residuals = projected.reshape(-1, 2) - pnp.img_pts
        rms = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
        pnp.draw_axes(image, estimator.K, estimator.D, 0.05)
        for point in pnp.original_img_pts:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 3, (0, 255, 0), 1)
        lines.extend([
            f"Board markers: {len(pnp.img_pts) // 4}/{len(estimator.config.board_marker_ids)} "
            f"| reprojection RMS: {rms:.2f} px",
            f"RGB camera XYZ in board (m): {position[0]:+.3f}, "
            f"{position[1]:+.3f}, {position[2]:+.3f}",
        ])
    for row, text in enumerate(lines):
        origin = (10, 25 + row * 25)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return image, result


def browse(bag, *, rotations=DEFAULT_ROTATIONS, start_frame=0, axis_limit=1.5,
           dictionary=None, board_scale=None, max_dimension=None):
    if max_dimension is not None and (
        not isinstance(max_dimension, int) or max_dimension <= 0
    ):
        raise ValueError("max_dimension must be a positive integer or None")
    with RealSenseReader(bag, rotations=rotations) as source:
        if not 0 <= start_frame < len(source):
            raise ValueError(f"start frame must be between 0 and {len(source) - 1}")
        board = None
        if dictionary is not None or board_scale is not None:
            board = letter_board(
                dictionary if dictionary is not None else DEFAULT_DICTIONARY,
                board_scale if board_scale is not None else DEFAULT_BOARD_SCALE,
            )
        estimator = make_estimator(source.calibration.color, board)
        board = estimator.config
        plotter = BoardPlotter3D(
            board, axis_limit=axis_limit, update_interval=1, camera_at_origin=False
        )
        window = "RGB camera / letter GridBoard"
        try:
            # Show both sides of the board plane using the existing board axes.
            plotter.ax.set_zlim(-axis_limit, axis_limit)
            plotter.ax.set_box_aspect((1, 1, 1))
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            print(f"Loaded {len(source)} frames; clockwise quarter-turns: {rotations}")
            print("Board marker size: "
                  f"{board.marker_length * 1000:.2f} mm; content width: "
                  f"{board.board_width * 1000:.2f} mm")
            print(PLAYBACK_HELP)
            print("RGB axes: X right, Y down, Z forward.")
            playback = Playback(source, start_frame)
            index, displayed = playback.index, None
            while plotter.plt.fignum_exists(plotter.fig.number):
                if index != displayed:
                    image, result = estimate_frame(source, index, estimator)
                    if result is not None:
                        # Pass board-to-camera unchanged: the existing plotter
                        # performs the inversion when camera_at_origin=False.
                        plotter.update(result[0])
                    else:
                        for artist in plotter.board_quivers:
                            artist.remove()
                        plotter.board_quivers.clear()
                    status = "pose available" if result is not None else "no pose"
                    plotter.ax.set_title(f"RGB camera in board frame | frame {index} | {status}")
                    plotter.fig.canvas.draw_idle()
                    cv2.imshow(window, _resize(image, max_dimension))
                    displayed = index
                plotter.fig.canvas.flush_events()
                playback.poll()
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                if playback.quit:
                    break
                index = playback.index
        finally:
            cv2.destroyAllWindows()
            plotter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=PLAYBACK_HELP)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--rotations", type=int, choices=range(4), default=DEFAULT_ROTATIONS,
                        help=f"clockwise quarter-turns applied by the reader (default: {DEFAULT_ROTATIONS})")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--dictionary", choices=("4x4", "5x5"),
                        help="override the configured BOARD with a letter board of this dictionary")
    parser.add_argument("--axis-limit", type=float, default=1.5,
                        help="3D plot extent in each direction, meters (default: 1.5)")
    parser.add_argument("--board-scale", type=float,
                        help=f"override letter board print scale (configured: {DEFAULT_BOARD_SCALE})")
    parser.add_argument("--max-dimension", type=int,
                        help="set RGB image's longer axis in pixels (default: native size)")
    args = parser.parse_args()
    if not args.bag.is_file():
        parser.error("bag must be an existing recording")
    if not np.isfinite(args.axis_limit) or args.axis_limit <= 0:
        parser.error("axis-limit must be finite and positive")
    if args.board_scale is not None and (not np.isfinite(args.board_scale) or args.board_scale <= 0):
        parser.error("board-scale must be finite and positive")
    if args.max_dimension is not None and args.max_dimension <= 0:
        parser.error("max-dimension must be positive")
    browse(args.bag, rotations=args.rotations, start_frame=args.start_frame,
           axis_limit=args.axis_limit, dictionary=args.dictionary,
           board_scale=args.board_scale, max_dimension=args.max_dimension)


if __name__ == "__main__":
    main()
