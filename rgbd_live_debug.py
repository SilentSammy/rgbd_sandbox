"""On-demand live RGB-D snapshots without resetting perception state."""

import json
import signal
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event

import cv2
import numpy as np


def _camera(camera):
    return dict(width=camera.width, height=camera.height, K=camera.K.tolist(),
                D=camera.D.tolist(), distortion_model=camera.distortion_model)


def _objects(result):
    tracks = {t.detection_index: t for t in result.tracks if t.visible}
    if result.segmentation is None:
        return []
    return [dict(
        label=obj.label,
        track_id=None if index not in tracks else tracks[index].id,
        confirmed=False if index not in tracks else tracks[index].confirmed,
        pixel_count=obj.pixel_count,
        voxel_count=obj.voxel_count,
        centroid_xyz=obj.centroid_xyz,
        min_bound_xyz=obj.min_bound_xyz,
        max_bound_xyz=obj.max_bound_xyz,
        extent_xyz=obj.extent_xyz,
        max_height_above_plane=obj.max_height_above_plane,
        centroid_depth_pixel=obj.centroid_depth_pixel,
        contour_depth_pixels=obj.contour_depth_pixels.tolist(),
    ) for index, obj in enumerate(result.segmentation.objects)]


class DebugCapture:
    def __init__(self, root, buffer_frames=15):
        self.root = Path(root)
        self.frames = deque(maxlen=buffer_frames)
        self.pending = Event()

    def install_signal(self):
        user_signal = getattr(signal, 'SIGUSR1', None)
        if user_signal is not None:
            signal.signal(user_signal, lambda *_: self.pending.set())
        return user_signal

    def append(self, frame):
        # RealSenseLive returns owned arrays, so retaining the Frame is sufficient.
        self.frames.append(frame)

    def consume_request(self):
        requested = self.pending.is_set()
        self.pending.clear()
        return requested

    def save(self, result, calibration, images, count, *, reason='manual', error=None):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        folder = self.root/f'{stamp}_frame_{count}_{reason}'
        folder.mkdir(parents=True)
        if result is not None:
            frame = result.frame
            cv2.imwrite(str(folder/'raw_rgb.png'), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
            np.save(folder/'depth_meters.npy', frame.depth)
            if images is not None:
                for name, image in zip(('annotated_rgb.png', 'annotated_depth.png'), images):
                    cv2.imwrite(str(folder/name), image)
            segmentation = result.segmentation
            report = dict(
                reason=reason, frame=count, timestamp_ns=frame.timestamp_ns,
                status=result.status, dt=result.dt, timings_ms=result.timings_ms,
                raw_pose=None if result.raw_pose is None else result.raw_pose[0].tolist(),
                filtered_pose=None if result.pose is None else result.pose[0].tolist(),
                plane_model=None if segmentation is None else segmentation.plane_model,
                segmentation_timings_ms=None if segmentation is None else segmentation.timings_ms,
                objects=_objects(result),
                tracks=[dict(id=t.id, position=t.position.tolist(), confirmed=t.confirmed,
                             visible=t.visible, detection_index=t.detection_index,
                             seen_seconds=t.seen_seconds, missing_seconds=t.missing_seconds)
                        for t in result.tracks],
                calibration=dict(color=_camera(calibration.color), depth=_camera(calibration.depth),
                                 depth_to_color=calibration.depth_to_color.tolist()),
                error=error,
            )
            (folder/'report.json').write_text(json.dumps(report, indent=2))
        if self.frames:
            np.savez(folder/'history.npz',
                     timestamps_ns=np.array([f.timestamp_ns for f in self.frames], np.int64),
                     color=np.stack([f.color for f in self.frames]),
                     depth=np.stack([f.depth for f in self.frames]))
        if error:
            (folder/'traceback.txt').write_text(error)
        return folder


object_details = _objects
