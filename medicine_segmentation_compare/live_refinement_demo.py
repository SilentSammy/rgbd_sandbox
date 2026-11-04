"""Fast depth IDs with asynchronous, pose-compensated SAM refinement."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('USE_TF', '0')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from async_refinement import AsyncRefiner, usable_refinement
from rgbd_geometry import reference_points_to_image
from rgbd_projection import Projector


def coarse_seeds(result, projector):
    """Project coarse depth labels once per frame, at segmentation resolution."""
    if result.segmentation is None:
        return []
    size = projector.depth_shape[::-1]
    depth = cv2.resize(result.frame.depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
    labels = cv2.resize(result.segmentation.object_labels, size, interpolation=cv2.INTER_NEAREST_EXACT)
    forward = projector.build_map(depth)
    z = projector.rays[2]*depth + projector.translation[2]
    seeds = []
    tracks = {t.detection_index: t for t in result.tracks if t.visible}
    for index, obj in enumerate(result.segmentation.objects):
        track = tracks.get(index)
        selected = (labels == obj.label) & np.isfinite(forward[:, :, 0])
        pixels = forward[selected]
        if len(pixels) < 10:
            continue
        lo, hi = np.floor(pixels.min(axis=0)).astype(int)-4, np.ceil(pixels.max(axis=0)).astype(int)+4
        h, w = result.frame.color.shape[:2]
        bounds = [max(0, int(lo[0])), max(0, int(lo[1])), min(w-1, int(hi[0])), min(h-1, int(hi[1]))]
        zs = z[selected]
        median = float(np.median(zs))
        band = float(max(.035, np.max(np.abs(np.percentile(zs, [5, 95])-median))+.025))
        seeds.append(dict(id=None if track is None else track.id, label=obj.label,
                          anchor=None if track is None else track.position.copy(), box=bounds,
                          recoveries=0 if track is None else getattr(track, 'recoveries', 0),
                          z=median, z_band=band, confirmed=track is not None and track.confirmed))
    return seeds


def projected_refinement(refined, track, seed, result, camera, max_age):
    if result.pose is None or not usable_refinement(refined, track, result.frame.timestamp_ns, max_age=max_age):
        return None, 'pose/identity/age/position'
    pixels, z = reference_points_to_image(refined.points, result.pose, camera)
    valid = np.isfinite(pixels).all(axis=1) & (z > 0)
    pixels, z = pixels[valid], z[valid]
    if len(pixels) < 20:
        return None, 'behind camera'
    if abs(float(np.median(z))-seed['z']) > max(.06, seed['z_band']):
        return None, 'current depth mismatch'
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    a, b, c, d = seed['box']
    intersection = max(0, min(hi[0], c)-max(lo[0], a))*max(0, min(hi[1], d)-max(lo[1], b))
    if intersection/max(1, min(np.prod(hi-lo), (c-a)*(d-b))) < .15:
        return None, 'current box mismatch'
    return [int(np.clip(lo[0], 0, camera.width-1)), int(np.clip(lo[1], 0, camera.height-1)),
            int(np.clip(hi[0], 0, camera.width-1)), int(np.clip(hi[1], 0, camera.height-1))], 'ok'


def draw(result, seeds, refined, camera, max_age, status):
    left = cv2.cvtColor(result.frame.color, cv2.COLOR_RGB2BGR)
    right = left.copy()
    tracks = {t.id: t for t in result.tracks}
    displayed = []
    for seed in seeds:
        identity = seed['id']
        x0, y0, x1, y1 = seed['box']
        text = f'Candidate {seed["label"]}' if identity is None else f'ID {identity}' + ('' if seed['confirmed'] else ' ?')
        cv2.rectangle(left, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(left, text, (x0, max(50, y0-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
        bounds, reason = None, 'waiting for SAM'
        if identity in refined:
            bounds, reason = projected_refinement(refined[identity], tracks[identity], seed, result, camera, max_age)
        age = None if identity not in refined else (result.frame.timestamp_ns-refined[identity].timestamp_ns)/1e9
        displayed.append(dict(id=identity, coarse_box=seed['box'], refined_box=bounds, reason=reason, age_seconds=age))
        if bounds is None:
            cv2.rectangle(right, (x0, y0), (x1, y1), (0, 255, 255), 1)
        else:
            x0, y0, x1, y1 = bounds
            cv2.rectangle(right, (x0, y0), (x1, y1), (0, 255, 0), 2)
            text += f' SAM {age:.1f}s'
        cv2.putText(right, text, (x0, max(50, y0-5)), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 255, 0) if bounds else (0, 255, 255), 1)
    for image, title in ((left, 'Depth: current coarse detections'), (right, 'Green: delayed SAM reprojected; yellow: coarse')):
        cv2.rectangle(image, (0, 0), (image.shape[1], 43), (0, 0, 0), -1)
        cv2.putText(image, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.putText(image, status, (6, 35), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
    return (left, right), displayed


def main():
    from contextlib import closing
    from rgbd_perception import PerceptionPipeline
    from rgbd_table_plane import BoardTabletopSegmenter
    from rgbd_tabletop_segmentation import TabletopSegmentationParameters
    from comparison import calibration_dict

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial')
    parser.add_argument('--bag', type=Path, help='optional recorded input instead of live camera')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-dimension', type=int, default=600)
    parser.add_argument('--model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--sam-threads', type=int, default=2)
    parser.add_argument('--refresh-seconds', type=float, default=2., help='minimum capture-time interval between requests')
    parser.add_argument('--max-age', type=float, default=15.)
    parser.add_argument('--scene-depth', type=float, default=2., help='maximum depth used for support-plane/object segmentation, meters')
    parser.add_argument('--plane-min-inliers', type=int, default=1000)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'outputs'/'refinement')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0)
    args = parser.parse_args()
    if min(args.max_dimension, args.sam_threads, args.max_age) <= 0 or args.refresh_seconds < 0 or args.frames < 0:
        parser.error('dimensions, threads and max-age must be positive; frame count/refresh nonnegative')
    cv2.setNumThreads(1)
    folder = args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (folder/'config.json').write_text(json.dumps(config, indent=2))
    snapshot = [False]
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, lambda *_: snapshot.__setitem__(0, True))
    if args.bag:
        from rgbd_db3_reader import RealSenseReader
        source = RealSenseReader(args.bag, rotations=1)
    else:
        from rgbd_live import RealSenseLive
        source = RealSenseLive(serial=args.serial)
    print(f'Outputs: {folder}\nC clear IDs/refinements | T relearn table | G snapshot | Q/Esc quit | PID {os.getpid()}', flush=True)
    worker = None
    try:
        with closing(source), (folder/'events.jsonl').open('a', buffering=1) as log:
            c = source.calibration
            (folder/'calibration.json').write_text(json.dumps(calibration_dict(c), indent=2))
            parameters = TabletopSegmentationParameters(max_depth=args.scene_depth, plane_min_inliers=args.plane_min_inliers)
            parameters.validate()
            perception = PerceptionPipeline(c, segmenter=BoardTabletopSegmenter(c, parameters))
            shape = (c.depth.height//2, c.depth.width//2)
            K = c.depth.K.copy()
            for row, factor in enumerate((shape[1]/c.depth.width, shape[0]/c.depth.height)):
                K[row] *= factor
                K[row, 2] += (factor-1)*.5
            projector = Projector(K, c.color.K, c.depth_to_color, shape,
                (c.color.height, c.color.width), depth_D=c.depth.D, color_D=c.color.D,
                depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)
            worker = AsyncRefiner(c, args.model, args.device, args.sam_threads, folder)
            refined, count, last_request, last_log = {}, 0, None, 0.
            windows = ('Depth coarse IDs', 'SAM refined IDs - live projection')
            if not args.headless:
                for i, name in enumerate(windows):
                    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow(name, i*370, 30)
            while not args.frames or count < args.frames:
                started = perf_counter()
                index = args.start_frame+count
                if args.bag and index >= len(source):
                    break
                frame = source[index] if args.bag else source.read()
                result = perception.process(frame)
                count += 1
                seeds = coarse_seeds(result, projector)
                message = worker.poll()
                if message:
                    if message['kind'] == 'result':
                        current_ids = {t.id for t in result.tracks}
                        for obj in message['objects']:
                            if obj.id in current_ids:
                                refined[obj.id] = obj
                        event = dict(kind='result', capture_timestamp_ns=message['timestamp_ns'],
                                     arrival_timestamp_ns=frame.timestamp_ns,
                                     latency_seconds=(frame.timestamp_ns-message['timestamp_ns'])/1e9,
                                     ids=[o.id for o in message['objects']], folder=message['folder'])
                    else:
                        event = message
                    print(event, flush=True)
                    log.write(json.dumps(event)+'\n')
                current_ids = {t.id for t in result.tracks}
                refined = {i: o for i, o in refined.items() if i in current_ids and (frame.timestamp_ns-o.timestamp_ns)/1e9 <= args.max_age}
                eligible = [s for s in seeds if s['confirmed']]
                if result.pose is not None and (last_request is None or (frame.timestamp_ns-last_request)/1e9 >= args.refresh_seconds):
                    if worker.submit(frame, result.pose, eligible):
                        last_request = frame.timestamp_ns
                status = f'Pose {"OK" if result.pose else "LOST"} | SAM {worker.state} | depth core {result.timings_ms["total"]:.0f} ms'
                status += ' | table '+perception.segmenter.status
                if result.segmentation_error:
                    status += ' | '+result.segmentation_error
                images, displayed = draw(result, seeds, refined, c.color, args.max_age, status)
                key = -1
                if not args.headless:
                    for name, image in zip(windows, images):
                        h, w = image.shape[:2]
                        scale = min(1, args.max_dimension/max(h, w))
                        cv2.imshow(name, cv2.resize(image, (round(w*scale), round(h*scale))))
                    key = cv2.waitKey(1) & 255
                    if key in (27, ord('q')) or any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
                        break
                    if key == ord('c'):
                        perception.persistor.clear()
                        refined.clear()
                    if key == ord('t'):
                        perception.segmenter.reset()
                summary = dict(frame=count, timestamp_ns=frame.timestamp_ns, pose_ok=result.pose is not None,
                               worker=worker.state, core_ms=result.timings_ms['total'],
                               loop_ms=(perf_counter()-started)*1000, objects=displayed,
                               retained_ids=list(refined), segmentation_error=result.segmentation_error,
                               raw_objects=0 if result.segmentation is None else len(result.segmentation.objects),
                               plane=None if result.segmentation is None else result.segmentation.plane_model,
                               plane_reference=None if perception.segmenter.reference_plane is None else perception.segmenter.reference_plane.tolist(),
                               plane_status=perception.segmenter.status, plane_support=perception.segmenter.support_count,
                               plane_residual_mm=perception.segmenter.residual_mm,
                               tracks=len(result.tracks))
                save = snapshot[0] or key == ord('g') or (args.headless and count == args.frames)
                if save:
                    target = folder/f'live_{frame.timestamp_ns}'
                    target.mkdir(exist_ok=True)
                    for name, image in zip(('coarse', 'refined'), images):
                        cv2.imwrite(str(target/f'{name}.png'), image)
                    cv2.imwrite(str(target/'rgb.png'), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
                    np.savez_compressed(target/'geometry.npz', depth=frame.depth,
                        **{f'points_{o.id}': o.points for o in refined.values()})
                    summary['reference_to_camera'] = None if result.pose is None else result.pose[0].tolist()
                    (target/'report.json').write_text(json.dumps(summary, indent=2))
                    snapshot[0] = False
                    print(f'Snapshot: {target}', flush=True)
                if perf_counter()-last_log >= 2:
                    log.write(json.dumps(summary)+'\n')
                    (folder/'latest.json').write_text(json.dumps(summary, indent=2))
                    print(f'{count}: {status} | loop {summary["loop_ms"]:.0f} ms | refined {sum(o["refined_box"] is not None for o in displayed)}/{len(seeds)}', flush=True)
                    last_log = perf_counter()
    finally:
        if worker is not None:
            worker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
