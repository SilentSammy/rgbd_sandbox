"""One asynchronous SAM request at a time; geometry stays in board coordinates."""

from dataclasses import dataclass
import multiprocessing as mp
from queue import Empty, Full
from time import perf_counter
import traceback

import cv2
import numpy as np


@dataclass
class RefinedObject:
    id: int
    timestamp_ns: int
    anchor: np.ndarray
    points: np.ndarray
    score: float


def mask_to_reference(mask, inverse, color_z, pose, camera, seed):
    """Measured visible surface only: z-buffer ownership and seed-depth gating.

    Honor the board estimator's implicit 180-degree pixel convention before
    unprojection; a bare optical-camera transform would use the wrong axes.
    """
    valid = mask & (inverse[:, :, 0] >= 0) & np.isfinite(color_z)
    valid &= np.abs(color_z-seed['z']) <= seed['z_band']
    y, x = np.where(valid)
    if len(x) < 20:
        return np.empty((0, 3))
    # Bound projection/rendering cost without changing the underlying mask.
    stride = max(1, int(np.ceil(len(x)/1500)))
    x, y = x[::stride], y[::stride]
    pixels = pose[1].to_image_points(np.column_stack((x, y)))
    rays = cv2.undistortPoints(pixels.astype(np.float64).reshape(-1, 1, 2), camera.K, camera.D).reshape(-1, 2)
    z = color_z[y, x]
    camera_points = np.column_stack((rays*z[:, None], z))
    transform = np.linalg.inv(pose[0])
    points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    return points[np.isfinite(points).all(axis=1)]


def usable_refinement(refined, track, timestamp_ns, *, max_age=15., max_shift=.08):
    """IDs are never recycled; reject moved/mismatched or expired geometry."""
    return (track.visible and track.confirmed and track.id == refined.id
            and 0 <= (timestamp_ns-refined.timestamp_ns)/1e9 <= max_age
            and np.linalg.norm(track.position-refined.anchor) <= max_shift)


def _worker(requests, responses, calibration, model, device, threads, output):
    # Imports/weights belong exclusively to this child process.
    from pathlib import Path
    import json
    from sam_backend import SamSegmenter
    from rgbd_projection import Projector
    from comparison import calibration_dict, overlay

    try:
        cv2.setNumThreads(1)
        sam = SamSegmenter(model, device)
        sam.torch.set_num_threads(threads)
        c = calibration
        projector = Projector(c.depth.K, c.color.K, c.depth_to_color,
            (c.depth.height, c.depth.width), (c.color.height, c.color.width),
            depth_D=c.depth.D, color_D=c.color.D, depth_model=c.depth.distortion_model,
            color_model=c.color.distortion_model)
        responses.put(dict(kind='ready', device=sam.device, load_ms=sam.load_ms))
    except Exception:
        responses.put(dict(kind='fatal', error=traceback.format_exc()))
        return
    while True:
        request = requests.get()
        if request is None:
            return
        try:
            started = perf_counter()
            frame, pose, seeds = request['frame'], request['pose'], request['seeds']
            prompts = [dict(kind='box', coordinates=seed['box']) for seed in seeds]
            masks, scores, timings = sam.segment(frame.color, prompts)
            _, inverse = projector.build_maps(frame.depth)
            valid = inverse[:, :, 0] >= 0
            x, y = inverse[valid].T
            z = projector.rays[2]*frame.depth + projector.translation[2]
            color_z = np.full(frame.color.shape[:2], np.nan, np.float32)
            color_z[valid] = z[y, x]
            objects, reports = [], []
            for seed, mask, score in zip(seeds, masks, scores):
                points = mask_to_reference(mask, inverse, color_z, pose, c.color, seed)
                x0, y0, x1, y1 = seed['box']
                inside = np.count_nonzero(mask[y0:y1+1, x0:x1+1])
                # Reject tiny label fragments or masks that escaped the proposal.
                fraction = inside/max(1, (x1-x0+1)*(y1-y0+1))
                accepted = score >= .5 and len(points) >= 20 and fraction >= .15
                if accepted:
                    objects.append(RefinedObject(seed['id'], frame.timestamp_ns,
                                                 seed['anchor'], points, score))
                reports.append(dict(id=seed['id'], box=seed['box'], score=score,
                                    measured_points=len(points), box_fill=fraction, accepted=accepted))
            compute_ms = (perf_counter()-started)*1000
            folder = Path(output)/f'request_{frame.timestamp_ns}'
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder/'rgb.png'), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(folder/'sam.png'), overlay(frame.color, masks, 'SAM capture frame; slots follow report IDs'))
            np.savez_compressed(folder/'capture.npz', depth=frame.depth, masks=masks,
                                reference_to_camera=pose[0], **{f'points_{o.id}': o.points for o in objects})
            report = dict(timestamp_ns=frame.timestamp_ns, objects=reports, timings_ms=timings,
                          compute_ms=compute_ms, total_worker_ms=(perf_counter()-started)*1000,
                          calibration=calibration_dict(c), image_size=pose[1].image_size,
                          seeds=[dict(s, anchor=s['anchor'].tolist()) for s in seeds])
            (folder/'report.json').write_text(json.dumps(report, indent=2))
            responses.put(dict(kind='result', timestamp_ns=frame.timestamp_ns, objects=objects,
                               report=report, folder=str(folder)))
        except Exception:
            responses.put(dict(kind='error', error=traceback.format_exc()))


class AsyncRefiner:
    """Never queue stale frames: submit the current frame only when idle."""

    def __init__(self, calibration, model, device, threads, output):
        context = mp.get_context('spawn')
        self.requests, self.responses = context.Queue(1), context.Queue(1)
        self.process = context.Process(target=_worker,
            args=(self.requests, self.responses, calibration, model, device, threads, str(output)), daemon=True)
        self.state, self.error = 'loading', ''
        self.process.start()

    def submit(self, frame, pose, seeds):
        if self.state != 'idle' or not seeds:
            return False
        self.requests.put_nowait(dict(frame=frame, pose=pose, seeds=seeds))
        self.state = 'busy'
        return True

    def poll(self):
        try:
            message = self.responses.get_nowait()
        except Empty:
            if not self.process.is_alive() and self.state != 'failed':
                self.state, self.error = 'failed', f'SAM process exited: {self.process.exitcode}'
                return dict(kind='fatal', error=self.error)
            return None
        self.state = 'failed' if message['kind'] == 'fatal' else 'idle'
        self.error = message.get('error', '')
        return message

    def close(self):
        if self.process.is_alive():
            try:
                self.requests.put_nowait(None)
            except Full:
                pass
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        self.requests.cancel_join_thread()
        self.requests.close()
        self.responses.close()
