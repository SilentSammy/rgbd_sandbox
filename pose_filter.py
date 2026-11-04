"""Short-window camera-pose smoothing, independent of images and pose provider."""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from rgbd_types import Pose


@dataclass
class Parameters:
    window: int = 3
    max_gap_seconds: float = .5


class PoseFilter:
    """Average camera positions and rotations in the fixed reference frame.

    update(Pose | None) -> Pose | None. Output retains the current capture time.
    Missing pose, reference changes, large capture gaps, repeated timestamps and
    playback direction changes reset history. No extrapolation or stale output.
    Parameters may change between calls; shrinking the window trims history.
    """

    def __init__(self, parameters=None):
        self.parameters = Parameters() if parameters is None else parameters
        self.reset()

    def reset(self):
        self.history = deque()
        self._timestamp = self._reference = None
        self._direction = 0

    def update(self, pose: Pose | None) -> Pose | None:
        p = self.parameters
        if not isinstance(p.window, int) or p.window < 1:
            raise ValueError("window must be a positive integer")
        if not np.isfinite(p.max_gap_seconds) or p.max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive")
        if pose is None:
            self.reset()
            return None
        if self._timestamp is not None:
            delta = (pose.timestamp_ns-self._timestamp)/1e9
            direction = int(np.sign(delta))
            if (pose.reference_frame != self._reference or not direction
                    or abs(delta) > p.max_gap_seconds or self._direction not in (0, direction)):
                self.reset()
            else:
                self._direction = direction
        self.history.append(pose.camera_to_reference.copy())
        while len(self.history) > p.window:
            self.history.popleft()
        self._timestamp, self._reference = pose.timestamp_ns, pose.reference_frame
        matrices = np.array(self.history)
        # Project the mean rotation onto SO(3), avoiding Euler-angle wraparound
        # and ensuring a proper rotation rather than an element-wise matrix mean.
        u, _, vt = np.linalg.svd(matrices[:, :3, :3].mean(axis=0))
        rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        camera_to_reference = np.eye(4)
        camera_to_reference[:3, :3] = rotation
        camera_to_reference[:3, 3] = matrices[:, :3, 3].mean(axis=0)
        return Pose(pose.timestamp_ns, np.linalg.inv(camera_to_reference), pose.reference_frame)


def main():
    import cv2
    from board_pose_estimator import Board, BoardPoseEstimator
    from rgbd_db3_reader import RealSenseReader
    from rgbd_demo import demo_parser, run_demo

    parser = demo_parser(__doc__)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-gap-seconds", type=float, default=.5)
    parser.add_argument("--board-scale", type=float, default=.96)
    args = parser.parse_args()
    with RealSenseReader(args.bag, rotations=args.rotations) as source:
        estimator = BoardPoseEstimator(source.calibration.color, Board(scale=args.board_scale))
        smoother = PoseFilter(Parameters(args.window, args.max_gap_seconds))
        previous_index = None

        def process(frame, index):
            nonlocal previous_index
            if previous_index is not None and abs(index-previous_index) != 1:
                smoother.reset()
            previous_index = index
            raw = estimator.estimate(frame)
            started = perf_counter()
            filtered = smoother.update(raw)
            filter_ms = (perf_counter()-started)*1000
            panes = []
            for label, pose in (("Raw", raw), ("Filtered", filtered)):
                pane = estimator.annotate(frame.color, pose)
                lines = [label, estimator.observation.status]
                if pose is not None:
                    lines.append("XYZ (m): "+", ".join(f"{v:+.3f}" for v in pose.camera_to_reference[:3, 3]))
                for row, line in enumerate(lines):
                    cv2.putText(pane, line, (8, 24+row*24), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(pane, line, (8, 24+row*24), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
                panes.append(pane)
            return np.hstack(panes), dict(status=estimator.observation.status, filter_ms=filter_ms,
                samples=len(smoother.history), raw_camera_to_reference=None if raw is None else raw.camera_to_reference.tolist(),
                filtered_camera_to_reference=None if filtered is None else filtered.camera_to_reference.tolist())

        run_demo(source, process, args, title="Raw | filtered camera pose")


if __name__ == "__main__":
    main()
