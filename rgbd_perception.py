"""Detection phase: source-independent geometric perception, without UI or OCR.

RGB -> BoardPoseEstimator -> PoseFilter -----+
Depth -> TabletopSegmenter -----------------+-> ObjectPersistor

One pipeline instance belongs to one calibrated stream and fixed reference.
Call process() serially with synchronized RGB-D frames; no frame queue is owned.
The filtered pose also anchors the default tabletop plane in reference space.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from board_pose_setup import make_pose_estimator
from board_pose_tracking import PoseFilter
from rgbd_geometry import object_positions_in_reference
from rgbd_object_persistence import ObjectMemory, Track
from rgbd_tabletop_segmentation import segment_tabletop_objects, TabletopSegmentationResult
from rgbd_table_plane import BoardTabletopSegmenter
from rgbd_types import Frame


class TabletopSegmenter:
    """Depth -> candidate-object masks and geometry in depth-camera coordinates."""

    def __init__(self, depth_camera, parameters=None, *, processing_scale=.5):
        self.camera, self.parameters = depth_camera, parameters
        self.processing_scale = processing_scale

    def segment(self, depth, *, pose=None):
        return segment_tabletop_objects(depth, self.camera, self.parameters,
                                       processing_scale=self.processing_scale)


class ObjectPersistor:
    """Objects + filtered pose + capture dt -> persistent track snapshots."""

    def __init__(self, calibration, memory=None, *, appearance=False):
        self.calibration = calibration
        self.appearance, self.appearance_ms = None, 0.
        if appearance:
            from rgbd_appearance import RGBObjectAppearance
            from rgbd_visual_memory import AppearanceMemory
            self.appearance = RGBObjectAppearance(calibration)
            self.memory = AppearanceMemory() if memory is None else memory
        else:
            self.memory = ObjectMemory() if memory is None else memory

    def update(self, objects, pose, dt, *, frame=None, segmentation=None):
        positions = [] if pose is None else object_positions_in_reference(objects, self.calibration, pose)
        started = perf_counter()
        observations = None
        if self.appearance is not None and pose is not None and frame is not None and segmentation is not None:
            observations = self.appearance.observe(frame, segmentation)
        self.appearance_ms = (perf_counter()-started)*1000
        tracks = self.memory.update(positions, dt, observations=observations)
        # Old results must not change when the next frame updates live memory.
        return tuple(replace(t, position=t.position.copy()) for t in tracks)

    def clear(self):
        self.memory.clear()

    def reset(self):
        self.clear()
        self.memory.elapsed_seconds = 0


@dataclass
class PerceptionResult:
    frame: Frame
    raw_pose: tuple | None  # (reference_to_camera 4x4, PnpResult), existing board convention.
    pose: tuple | None  # Same contract, after temporal filtering.
    segmentation: TabletopSegmentationResult | None
    tracks: tuple[Track, ...]  # Positions in fixed reference coordinates, meters.
    dt: float
    pose_status: str
    segmentation_error: str
    timings_ms: dict[str, float]

    @property
    def status(self):
        return self.segmentation_error or self.pose_status


class PerceptionPipeline:
    """Coordinate independent stages; timestamps, not processing time, age tracks.

    Live mode requires increasing capture timestamps. allow_reverse=True is
    for bag browsing only: dt becomes abs(timestamp difference), as before.
    Optional frame_index resets geometry on jumps/reversals, but not memory.
    Stage instances can be supplied to change one component without changing UI.
    """

    def __init__(self, calibration, *, board=None, pose_estimator=None, pose_filter=None,
                 segmenter=None, persistor=None, allow_reverse=False):
        self.calibration = calibration
        self.pose_estimator = make_pose_estimator(calibration.color, board) if pose_estimator is None else pose_estimator
        self.pose_filter = PoseFilter() if pose_filter is None else pose_filter
        self.segmenter = BoardTabletopSegmenter(calibration) if segmenter is None else segmenter
        self.persistor = ObjectPersistor(calibration, appearance=True) if persistor is None else persistor
        self.allow_reverse = allow_reverse
        self.previous_timestamp = self.last_index = None
        self.direction = 0

    def reset_geometry(self):
        self.pose_estimator.reset()
        self.pose_filter.reset()
        if hasattr(self.segmenter, 'reset'):
            self.segmenter.reset()
        self.last_index, self.direction = None, 0

    def reset(self):
        """Start a new stream/session; IDs remain non-recycled within this instance."""
        self.reset_geometry()
        self.persistor.reset()
        self.previous_timestamp = None

    def process(self, frame, *, frame_index=None):
        c = self.calibration
        if not isinstance(frame.timestamp_ns, (int, np.integer)):
            raise ValueError("Capture timestamp must be an integer in nanoseconds")
        if frame.color.shape != (c.color.height, c.color.width, 3) or frame.depth.shape != (c.depth.height, c.depth.width):
            raise ValueError("Frame dimensions must match pipeline calibration")
        if frame.color.dtype != np.uint8 or frame.depth.dtype.kind != "f":
            raise ValueError("Require RGB uint8 and floating-point depth in meters")
        delta_ns = 0 if self.previous_timestamp is None else frame.timestamp_ns-self.previous_timestamp
        if not self.allow_reverse and self.previous_timestamp is not None and delta_ns <= 0:
            raise ValueError("Live capture timestamps must increase; use allow_reverse=True for playback")
        dt = abs(delta_ns)/1e9
        started = perf_counter()
        if frame_index is not None:
            if self.last_index is not None:
                delta = frame_index-self.last_index
                direction = int(np.sign(delta))
                if abs(delta) != 1 or self.direction not in (0, direction):
                    self.reset_geometry()
                self.direction = direction if abs(delta) == 1 else 0
            self.last_index = frame_index
        raw_pose = self.pose_estimator.estimate(frame.color, frame_index=frame_index)
        pose_ms = (perf_counter()-started)*1000
        stage = perf_counter()
        pose = self.pose_filter.update(raw_pose)
        filter_ms = (perf_counter()-stage)*1000
        stage = perf_counter()
        error = ""
        try:
            segmentation = self.segmenter.segment(frame.depth, pose=pose)
        except (RuntimeError, ValueError) as exc:
            segmentation, error = None, str(exc)
        segmentation_ms = (perf_counter()-stage)*1000
        stage = perf_counter()
        objects = () if segmentation is None else segmentation.objects
        tracks = self.persistor.update(objects, pose, dt, frame=frame, segmentation=segmentation)
        persistence_ms = (perf_counter()-stage)*1000
        self.previous_timestamp = frame.timestamp_ns
        status = self.pose_estimator.status
        if pose is not None:
            status += " | filtered"
        timings = dict(pose=pose_ms, filter=filter_ms, segmentation=segmentation_ms,
                       persistence=persistence_ms, total=(perf_counter()-started)*1000)
        return PerceptionResult(frame, raw_pose, pose, segmentation, tracks, dt, status, error, timings)
