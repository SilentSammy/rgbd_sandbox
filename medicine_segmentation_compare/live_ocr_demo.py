"""Persistent depth objects + background OCR, with optional background SAM.

Defaults to a recording. --live explicitly opens the robot camera.
"""

import argparse
from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter, sleep
import unicodedata

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('USE_TF', '0')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from async_refinement import AsyncRefiner
from comparison import SnapshotSource, calibration_dict
from live_refinement_demo import coarse_seeds, draw
from rgbd_db3_viewer import Playback, PLAYBACK_HELP
from rgbd_object_ocr import ObjectOCR
from rgbd_object_persistence import ObjectMemory
from rgbd_visual_memory import AppearanceMemory
from rgbd_perception import PerceptionPipeline, ObjectPersistor
from rgbd_table_plane import BoardTabletopSegmenter
from rgbd_tabletop_segmentation import TabletopSegmentationParameters
from rgbd_types import Camera, Calibration, Frame


class CapturedSequence:
    """Replay the diagnostic capture without opening a camera."""
    def __init__(self, folder):
        c = json.loads((folder/'calibration.json').read_text())
        def camera(v):
            return Camera(v['width'], v['height'], np.array(v['K']), np.array(v['D']), v['distortion_model'])
        self.calibration = Calibration(camera(c['color']), camera(c['depth']), np.array(c['depth_to_color']))
        with np.load(folder/'sequence.npz') as data:
            self.rgb, self.depth, self.timestamps = data['rgb'], data['depth'], data['timestamp_ns']
        self.fps = 1e9/np.median(np.diff(self.timestamps)) if len(self.timestamps) > 1 else 30

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        return Frame(int(self.timestamps[index]), self.rgb[index], self.depth[index])

    def close(self):
        pass


def label_panel(seeds, ocr, elapsed, size):
    panel = np.full((max(600, 110+80*len(seeds)), 600, 3), 25, np.uint8)
    def text(value, y, color=(230, 230, 230)):
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()
        cv2.putText(panel, value[:78], (12, y), cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
    timing = '' if ocr.last_ms is None else f' | last batch {ocr.last_ms:.0f} ms'
    text('OCR '+ocr.state+timing, 24)
    text('R reread | C clear IDs | T relearn table | G snapshot', 46)
    text('Dictionary suggestions; raw OCR retained below each ID', 68)
    if ocr.error:
        text(ocr.error.strip().splitlines()[-1], 90, (0, 100, 255))
    for row, seed in enumerate(seeds):
        y = 120+row*80
        item = ocr.cache.get(seed['id'])
        name = 'probation' if not seed['confirmed'] else 'waiting / unknown'
        if item:
            name = item['label'] or ('AMBIGUOUS' if item['latest_ambiguous'] else 'Unknown')
            if item['label_remembered']:
                name += ' (remembered)'
            if item['label_conflict']:
                name += ' [conflict]'
        recovered = f' [R{seed["recoveries"]}]' if seed.get('recoveries', 0) else ''
        text(f'ID {seed["id"] if seed["id"] is not None else "?"}{recovered}: {name}', y, (0, 255, 255))
        if item:
            raw = ' | '.join(f'{t["text"]} ({t["confidence"]:.2f})' for t in item['raw'][:3])
            text('Raw: '+(raw or '(no readable text)'), y+21)
            best = item['best_match']
            if best:
                text(f'Best similarity {best["score"]:.2f}, OCR {best["ocr_confidence"]:.2f} | age {max(0, elapsed-best["elapsed_seconds"]):.1f}s', y+42)
            else:
                text('No remembered dictionary match', y+42)
            latest = 'AMBIGUOUS' if item['latest_ambiguous'] else (item['latest_label'] or 'Unknown')
            text(f'Latest: {latest} | read age {max(0, elapsed-item["elapsed_seconds"]):.1f}s', y+62)
    scale = min(1, size/max(panel.shape[:2]))
    return cv2.resize(panel, (round(panel.shape[1]*scale), round(panel.shape[0]*scale)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=PLAYBACK_HELP)
    parser.add_argument('source', nargs='?', type=Path, default=ROOT/'recordings/20261005_195417.db3')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--serial')
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--max-dimension', type=int, default=600)
    parser.add_argument('--scene-depth', type=float, default=2.)
    parser.add_argument('--plane-min-inliers', type=int, default=1000)
    parser.add_argument('--max-footprint', type=float, default=.20, help='maximum tabletop footprint diameter in meters; 0 disables')
    parser.add_argument('--probation-seconds', type=float, help='default: 1s live, 0s for offline inspection')
    parser.add_argument('--no-appearance', action='store_true', help='position-only baseline for comparison')
    parser.add_argument('--forget-after', type=float, default=30., help='retain missing confirmed IDs for this many capture seconds')
    parser.add_argument('--ocr-refresh-seconds', type=float, default=5.)
    parser.add_argument('--ocr-scale', type=float, default=2.)
    parser.add_argument('--ocr-threads', type=int, default=2)
    parser.add_argument('--sam', action='store_true', help='also refine boxes with the existing SAM worker')
    parser.add_argument('--model', default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--wait-seconds', type=float, default=60., help='headless worker-drain timeout after the final frame')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'outputs'/'ocr')
    args = parser.parse_args()
    if min(args.max_dimension, args.ocr_scale, args.ocr_threads, args.wait_seconds) <= 0 or args.ocr_refresh_seconds < 0 or args.frames < 0:
        parser.error('dimensions, scale, threads and timeout must be positive; frame count/refresh nonnegative')
    if args.max_footprint < 0:
        parser.error('max-footprint must be nonnegative')
    if args.probation_seconds is not None and args.probation_seconds < 0:
        parser.error('probation must be nonnegative')
    cv2.setNumThreads(1)
    if args.live:
        from rgbd_live import RealSenseLive
        source = RealSenseLive(serial=args.serial)
    elif args.source.is_dir():
        source = CapturedSequence(args.source) if (args.source/'sequence.npz').exists() else SnapshotSource(args.source)
    else:
        from rgbd_db3_reader import RealSenseReader
        source = RealSenseReader(args.source, rotations=1)
    folder = args.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    (folder/'config.json').write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    save_requested = [False]
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, lambda *_: save_requested.__setitem__(0, True))
    ocr = sam = None
    print(f'Outputs: {folder}', flush=True)
    print(PLAYBACK_HELP+' | R reread | C clear IDs | T table | G snapshot', flush=True)
    try:
        with closing(source), (folder/'events.jsonl').open('a', encoding='utf-8', buffering=1) as log:
            c = source.calibration
            (folder/'calibration.json').write_text(json.dumps(calibration_dict(c), indent=2))
            parameters = TabletopSegmentationParameters(max_depth=args.scene_depth, plane_min_inliers=args.plane_min_inliers,
                                                        max_object_footprint=args.max_footprint or None)
            parameters.validate()
            probation = args.probation_seconds if args.probation_seconds is not None else (1. if args.live else 0.)
            memory_type = ObjectMemory if args.no_appearance else AppearanceMemory
            memory = memory_type(probation_period=probation, forget_after=args.forget_after)
            pipeline = PerceptionPipeline(c, segmenter=BoardTabletopSegmenter(c, parameters),
                persistor=ObjectPersistor(c, memory, appearance=not args.no_appearance), allow_reverse=not args.live)
            # Same calibrated half-resolution display projection as the SAM demo.
            from rgbd_projection import Projector
            shape = c.depth.height//2, c.depth.width//2
            K = c.depth.K.copy()
            for row, factor in enumerate((shape[1]/c.depth.width, shape[0]/c.depth.height)):
                K[row] *= factor
                K[row, 2] += (factor-1)*.5
            projector = Projector(K, c.color.K, c.depth_to_color, shape, (c.color.height, c.color.width),
                depth_D=c.depth.D, color_D=c.color.D, depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)
            ocr = ObjectOCR(folder, refresh_seconds=args.ocr_refresh_seconds, scale=args.ocr_scale, threads=args.ocr_threads)
            if args.sam:
                sam = AsyncRefiner(c, args.model, 'auto', 2, folder/'sam')
            start = args.start_frame if args.start_frame is not None else (137 if not args.live and not args.source.is_dir() else 0)
            playback = None if args.live else Playback(source, start)
            index, previous, result, count = start, None, None, 0
            refined, last_sam, sam_received = {}, None, False
            finishing, deadline, paused, force_ocr, last_log = False, None, False, False, 0.
            windows = ('Objects + OCR IDs', 'Object label readings')
            positioned = False
            if not args.headless:
                for i, window in enumerate(windows):
                    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow(window, 370*i, 30)
            while True:
                started = perf_counter()
                if not finishing and ((args.live and not paused) or result is None or (not args.live and index != previous)):
                    frame = source.read() if args.live else source[index]
                    result = pipeline.process(frame, frame_index=None if args.live else index)
                    seeds = coarse_seeds(result, projector)
                    for event in getattr(memory, 'events', []):
                        log.write(json.dumps(dict(event, timestamp_ns=int(frame.timestamp_ns)))+'\n')
                        if event['kind'] == 'appearance_recovery':
                            print(event, flush=True)
                    count += 1
                    previous = index
                    if args.headless and ((args.frames and count >= args.frames) or (not args.live and index == len(source)-1)):
                        finishing, deadline = True, perf_counter()+args.wait_seconds
                elapsed = pipeline.persistor.memory.elapsed_seconds
                message = ocr.poll(result.tracks)
                if message:
                    print(message, flush=True)
                    log.write(json.dumps(message, ensure_ascii=False)+'\n')
                    if hasattr(memory, 'remember_label'):
                        for identity, reading in ocr.cache.items():
                            memory.remember_label(identity, reading['best_match'], conflict=reading['label_conflict'])
                if sam is not None:
                    message = sam.poll()
                    if message:
                        if message['kind'] == 'result':
                            ids = {t.id for t in result.tracks}
                            refined.update({o.id: o for o in message['objects'] if o.id in ids})
                            sam_received, force_ocr = True, True
                        else:
                            print(message, flush=True)
                    if result.pose is not None and (last_sam is None or elapsed-last_sam >= 5):
                        if sam.submit(result.frame, result.pose, [s for s in seeds if s['confirmed']]):
                            last_sam = elapsed
                ids = {t.id for t in result.tracks}
                refined = {i: o for i, o in refined.items() if i in ids}
                status = f'Pose {"OK" if result.pose else "LOST"} | table {pipeline.segmenter.status} | OCR {ocr.state}'
                if sam is not None:
                    status += ' | SAM '+sam.state
                images, displayed = draw(result, seeds, refined, c.color, 15., status)
                boxes = {o['id']: o['refined_box'] for o in displayed if o['refined_box'] is not None}
                requests = []
                for seed in seeds:
                    bounds = seed['box']
                    if seed['id'] in boxes:
                        other = boxes[seed['id']]
                        # Measured SAM geometry may miss text near depth holes.
                        bounds = [min(bounds[0], other[0]), min(bounds[1], other[1]),
                                  max(bounds[2], other[2]), max(bounds[3], other[3])]
                    requests.append(dict(seed, box=bounds, box_source='depth+sam' if seed['id'] in boxes else 'depth'))
                if ocr.submit(result.frame, requests, elapsed, force=force_ocr):
                    force_ocr = False
                image = images[1] if args.sam else images[0]
                panel = label_panel(seeds, ocr, elapsed, args.max_dimension)
                h, w = image.shape[:2]
                scale = min(1, args.max_dimension/max(h, w))
                view = cv2.resize(image, (round(w*scale), round(h*scale)))
                key = -1
                if not args.headless:
                    cv2.imshow(windows[0], view)
                    cv2.imshow(windows[1], panel)
                    if not positioned:
                        cv2.waitKey(1)
                        cv2.moveWindow(windows[0], 0, 30)
                        cv2.moveWindow(windows[1], 390, 30)
                        positioned = True
                    key = cv2.waitKey(1 if args.live else 15) & 255
                    if key == 27 or (args.live and key == ord('q')) or any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
                        break
                    if args.live and key == ord('s'):
                        paused = not paused
                    if key == ord('c'):
                        pipeline.persistor.clear()
                        ocr.clear()
                        refined.clear()
                        previous, result = None, None
                    if key == ord('t'):
                        pipeline.segmenter.reset()
                        previous, result = None, None
                    if key == ord('r'):
                        force_ocr = True
                summary = dict(frames=count, source_index=index, timestamp_ns=int(frame.timestamp_ns),
                    elapsed_seconds=elapsed, core_ms=result.timings_ms['total'] if result else None,
                    loop_ms=(perf_counter()-started)*1000, status=status,
                    segmentation_error='' if result is None else result.segmentation_error,
                    objects=displayed, ocr_state=ocr.state, ocr_jobs=ocr.completed_jobs, ocr=list(ocr.cache.values()))
                summary['sam_state'] = None if sam is None else sam.state
                summary['sam_error'] = '' if sam is None else sam.error
                summary['ocr_error'] = ocr.error
                summary['appearance_ms'] = pipeline.persistor.appearance_ms
                summary['plane'] = None if result is None or result.segmentation is None else result.segmentation.plane_model
                summary['detections'] = [] if result is None or result.segmentation is None else [
                    dict(label=o.label, pixels=o.pixel_count, centroid_xyz=o.centroid_xyz,
                         extent_xyz=o.extent_xyz, footprint_m=o.footprint_diameter)
                    for o in result.segmentation.objects]
                summary['tracks'] = [dict(id=t.id, visible=t.visible, confirmed=t.confirmed,
                    detection_index=t.detection_index,
                    position=t.position.tolist(), missing_seconds=t.missing_seconds,
                    association=getattr(t, 'association', 'position'), recoveries=getattr(t, 'recoveries', 0),
                    label=getattr(t, 'label', None), templates=len(getattr(t, 'templates', ()))) for t in result.tracks] if result else []
                no_candidates = not any(s['confirmed'] for s in seeds)
                done = finishing and (ocr.state == 'failed' or no_candidates or
                    (ocr.completed_jobs > 0 and ocr.state == 'idle' and (sam is None or sam_received or sam.state == 'failed'))
                    or perf_counter() >= deadline)
                if save_requested[0] or key == ord('g') or done:
                    target = folder/f'view_{count:05d}'
                    target.mkdir(exist_ok=True)
                    cv2.imwrite(str(target/'objects.png'), view)
                    cv2.imwrite(str(target/'labels.png'), panel)
                    cv2.imwrite(str(target/'raw_rgb.png'), cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR))
                    np.save(target/'depth_meters.npy', frame.depth)
                    summary['calibration'] = calibration_dict(c)
                    summary['reference_to_camera'] = None if result is None or result.pose is None else result.pose[0].tolist()
                    (target/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
                    save_requested[0] = False
                if perf_counter()-last_log >= 2 or done:
                    (folder/'latest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
                    log.write(json.dumps(summary, ensure_ascii=False)+'\n')
                    last_log = perf_counter()
                if done:
                    if ocr.state == 'failed' or (not no_candidates and not ocr.completed_jobs):
                        raise RuntimeError(ocr.error or 'OCR did not complete before the worker timeout')
                    if sam is not None and sam.state == 'failed':
                        raise RuntimeError(sam.error)
                    break
                if not args.live and not finishing:
                    if args.headless:
                        index += 1
                    else:
                        playback.update(key)
                        if playback.quit:
                            break
                        index = playback.index
                if finishing:
                    sleep(.01)
    finally:
        if sam is not None:
            sam.close()
        if ocr is not None:
            ocr.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
