"""RGB-only GridBoard optical-flow fallback and short-window camera pose filtering.

Adapted from CornerTracker/PoseTracker in the supplied puzzlebot project.
Tracking is per marker ID; hidden markers are not inferred from other markers.
"""

from collections import deque
import copy
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from board_est import BoardEstimator


class PoseWindow:
    """Mean of the latest camera-in-board poses, with proper rotation averaging."""

    def __init__(self, size=3):
        self.poses = deque(maxlen=size)

    def update(self, board_to_camera):
        self.poses.append(np.linalg.inv(board_to_camera))
        poses = np.array(self.poses)
        camera_to_board = np.eye(4)
        camera_to_board[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix()
        camera_to_board[:3, 3] = poses[:, :3, 3].mean(axis=0)
        return np.linalg.inv(camera_to_board)


def _patch(gray, corners):
    square = np.float32([[0, 0], [31, 0], [31, 31], [0, 31]])
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), square)
    image = cv2.warpPerspective(gray, transform, (32, 32)).astype(np.float32)
    return (image-image.mean()) / max(float(image.std()), 1), float(image.std())


@dataclass
class MarkerTrack:
    corners: np.ndarray
    anchors: np.ndarray
    points: np.ndarray
    template: np.ndarray
    last_corners: np.ndarray
    age: int = 0


class TrackedBoard:
    """Use decoded corners first; recover missing IDs only from their own texture.

    Batch forward/backward LK for missing markers, then fit individual anchored
    homographies. Reject poor flow, shape changes, or mismatched marker patches.
    A fallback lasts at most three visited frames before fresh decoding is needed.
    """

    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def __init__(self, config, max_tracked_frames=3):
        self.config = config
        self.max_tracked_frames = max_tracked_frames
        self.reset()

    def __getattr__(self, name):
        return getattr(self.config, name)

    def reset(self):
        self.previous_gray = None
        self.tracks = {}
        self.detected_ids = self.tracked_ids = ()

    def _seed(self, gray, corners):
        x, y, w, h = cv2.boundingRect(corners)
        x0, y0 = max(0, x-2), max(0, y-2)
        crop = gray[y0:min(gray.shape[0], y+h+2), x0:min(gray.shape[1], x+w+2)]
        mask = np.zeros(crop.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(corners-[x0, y0]).astype(np.int32), 255)
        points = cv2.goodFeaturesToTrack(crop, 32, 0.01, 2, mask=mask)
        if points is None or len(points) < 6:
            return None
        points += np.float32([x0, y0])
        template, _ = _patch(gray, corners)
        return MarkerTrack(corners.copy(), points.copy(), points, template, corners.copy())

    def _recover(self, gray, track, points, good):
        if good.sum() < 6:
            return None
        anchors, points = track.anchors[good], points[good]
        transform, inliers = cv2.findHomography(anchors, points, cv2.RANSAC, 1.5)
        if transform is None or inliers.sum() < 6 or inliers.mean() < 0.75:
            return None
        corners = cv2.perspectiveTransform(track.corners.reshape(-1, 1, 2), transform).reshape(4, 2)
        if (not np.isfinite(corners).all() or not cv2.isContourConvex(corners)
                or np.any(corners < 0) or np.any(corners >= [gray.shape[1], gray.shape[0]])):
            return None
        ratio = abs(cv2.contourArea(corners)) / max(abs(cv2.contourArea(track.last_corners)), 1)
        if not 0.75 <= ratio <= 1.33:
            return None
        edges = np.roll(corners, -1, axis=0)-corners
        cosines = np.sum(-np.roll(edges, 1, axis=0)*edges, axis=1)
        cosines /= np.maximum(np.linalg.norm(edges, axis=1)*np.linalg.norm(np.roll(edges, 1, axis=0), axis=1), 1e-6)
        if np.any(cosines > np.cos(np.deg2rad(30))):
            return None
        patch, contrast = _patch(gray, corners)
        if contrast < 23 or np.mean(patch*track.template) < 0.7:
            return None
        keep = inliers.ravel().astype(bool)
        track.anchors, track.points = anchors[keep], points[keep]
        track.last_corners = corners
        track.age += 1
        return corners

    def detect_corners(self, frame, drawing_frame=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self.config.detect_corners(frame)
        found = {} if ids is None else {int(i): c.reshape(4, 2).astype(np.float32)
                                       for i, c in zip(ids.ravel(), corners) if int(i) in self.board_marker_ids}
        self.detected_ids = tuple(sorted(found))
        missing = {i: t for i, t in self.tracks.items()
                   if i not in found and t.age < self.max_tracked_frames}
        recovered = {}
        if missing and self.previous_gray is not None:
            points = np.concatenate([t.points for t in missing.values()])
            new, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, points, None, **self.LK)
            if new is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, new, None, **self.LK)
                if back is not None:
                    good = (status.ravel() != 0) & (back_status.ravel() != 0)
                    good &= np.linalg.norm((points-back).reshape(-1, 2), axis=1) < 1
                    offset = 0
                    for i, track in missing.items():
                        stop = offset+len(track.points)
                        quad = self._recover(gray, track, new[offset:stop], good[offset:stop])
                        if quad is not None:
                            recovered[i] = quad
                        offset = stop
        self.tracks = {i: missing[i] for i in recovered}
        for i, quad in found.items():
            track = self._seed(gray, quad)
            if track is not None:
                self.tracks[i] = track
        self.previous_gray = gray
        self.tracked_ids = tuple(sorted(recovered))
        found.update(recovered)
        if not found:
            return (), None
        ordered = sorted(found)
        corners = [found[i].reshape(1, 4, 2) for i in ordered]
        ids = np.array(ordered, dtype=np.int32).reshape(-1, 1)
        if drawing_frame is not None:
            cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
        return corners, ids


class BoardPoseTracker:
    """Drop-in RGB estimator; preserve board axes and synchronize filtered PnP."""

    def __init__(self, config, K, D, *, window=3, max_tracked_frames=3):
        self.estimator = BoardEstimator(TrackedBoard(config, max_tracked_frames), K, D)
        self.filter = PoseWindow(window)
        self.reset()

    def __getattr__(self, name):
        return getattr(self.estimator, name)

    def reset(self):
        self.filter.poses.clear()
        self.estimator.config.reset()
        self.last_index = None
        self.direction = 0
        self.raw_pose = None
        self.status = "NO BOARD POSE"

    def get_board_transform(self, frame, drawing_frame=None, *, frame_index=None):
        if frame_index is not None:
            if self.last_index is not None:
                delta = frame_index-self.last_index
                direction = int(np.sign(delta))
                if abs(delta) != 1 or self.direction not in (0, direction):
                    self.reset()
                self.direction = direction if abs(delta) == 1 else 0
            self.last_index = frame_index
        self.raw_pose = self.estimator.get_board_transform(frame)
        if self.raw_pose is None:
            self.filter.poses.clear()
            self.status = "NO BOARD POSE"
            return None
        transform, raw_pnp = self.raw_pose
        transform = self.filter.update(transform)
        pnp = copy.copy(raw_pnp)
        # Public board axes include the existing X half-turn. PnP uses its
        # pre-convention rotation; annotations and downstream pose must agree.
        rotation = transform[:3, :3] @ np.diag([1, -1, -1])
        pnp.rvec = cv2.Rodrigues(rotation)[0].ravel()
        pnp.tvec = transform[:3, 3].copy()
        config = self.estimator.config
        self.status = (f"RGB pose: {len(config.detected_ids)} detected, {len(config.tracked_ids)} tracked"
                       f" | filter {len(self.filter.poses)}/{self.filter.poses.maxlen}")
        if drawing_frame is not None:
            pnp.draw_axes(drawing_frame, self.K, self.D)
        return transform, pnp
