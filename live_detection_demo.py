"""Live RGB-D detection with persistent IDs. C clears memory; Esc/Q exits."""

import argparse
import json
import os
import traceback
from collections import deque
from pathlib import Path
from time import perf_counter

# Small frames run more predictably without nested BLAS/OpenMP thread pools.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import cv2
import numpy as np

from rgbd_live import RealSenseLive
from rgbd_live_debug import DebugCapture, object_details
from rgbd_perception import PerceptionPipeline
from rgbd_table_plane import BoardTabletopSegmenter
from rgbd_projection import Projector


def draw(result, projector, max_dimension):
    frame = result.frame
    color = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
    depth = cv2.applyColorMap(np.clip(frame.depth/2*255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth[frame.depth <= 0] = 0
    if result.segmentation is not None:
        size = projector.depth_shape[1], projector.depth_shape[0]
        projected_depth = cv2.resize(frame.depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
        labels = cv2.resize(result.segmentation.object_labels, size, interpolation=cv2.INTER_NEAREST_EXACT)
        mapping = projector.build_map(projected_depth)
        tracks = {t.detection_index: t for t in result.tracks if t.visible}
        for index, obj in enumerate(result.segmentation.objects):
            track = tracks.get(index)
            identity = obj.label if track is None else track.id
            tint = tuple(int(v) for v in np.random.default_rng(identity).integers(70, 256, 3))
            if track is not None and not track.confirmed:
                tint = (0, 255, 255)
            text = f'Candidate {obj.label}' if track is None else f'ID {track.id}' + ('' if track.confirmed else ' ?')
            mask = labels == obj.label
            projected = mapping[mask]
            projected = projected[np.isfinite(projected).all(axis=1)]
            contour = None
            if len(projected):
                x, y, w, h = cv2.boundingRect(np.rint(projected).astype(np.int32))
                contour = np.int32([[x,y], [x+w-1,y], [x+w-1,y+h-1], [x,y+h-1]])
            for image, points in ((color, contour), (depth, obj.contour_depth_pixels)):
                if points is not None:
                    cv2.polylines(image, [points], True, tint, 2)
                    x, y = map(int, points[0])
                    cv2.putText(image, text,
                                (x, max(20, y-6)), cv2.FONT_HERSHEY_SIMPLEX, .5, tint, 1)
    images = []
    for image in (color, depth):
        h, w = image.shape[:2]
        scale = min(1, max_dimension/max(h,w))
        image = cv2.resize(image, (round(w*scale), round(h*scale)))
        cv2.putText(image, result.status, (8,22), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
        images.append(image)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--width', type=int, default=848)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--rotations', type=int, choices=range(4), default=1)
    parser.add_argument('--segmentation-scale', type=float, choices=(.5,1), default=.5)
    parser.add_argument('--max-dimension', type=int, default=600)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int, default=0, help='stop after N processed frames; 0 runs continuously')
    parser.add_argument('--output-dir', type=Path, help='save last frame annotations and timing report')
    parser.add_argument('--debug-dir', type=Path, default=Path('debug_snapshots'))
    parser.add_argument('--debug-buffer-frames', type=int, default=15)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.max_dimension <= 0 or args.frames < 0 or args.debug_buffer_frames < 0:
        parser.error('dimensions must be positive and frame counts nonnegative')
    debug = DebugCapture(args.debug_dir, args.debug_buffer_frames)
    user_signal = debug.install_signal()
    timings, rendering, loops = (deque(maxlen=600) for _ in range(3))
    count = 0
    result = images = None
    windows = ('Live RGB detection', 'Live depth detection')
    with RealSenseLive(serial=args.serial, rotations=args.rotations, width=args.width,
                      height=args.height, fps=args.fps) as source:
        c = source.calibration
        perception = PerceptionPipeline(c, segmenter=BoardTabletopSegmenter(c, processing_scale=args.segmentation_scale))
        depth_shape = (c.depth.height//2,c.depth.width//2) if args.segmentation_scale == .5 else (c.depth.height,c.depth.width)
        depth_K = c.depth.K.copy()
        for row, factor in enumerate((depth_shape[1]/c.depth.width, depth_shape[0]/c.depth.height)):
            depth_K[row] *= factor
            depth_K[row,2] += (factor-1)*.5
        projector = Projector(depth_K, c.color.K, c.depth_to_color,
                              depth_shape, (c.color.height,c.color.width),
                              depth_D=c.depth.D, color_D=c.color.D,
                              depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)
        remote = '' if user_signal is None else f' | SSH snapshot: kill -USR1 {os.getpid()}'
        print(f'{source.name} {source.serial} | capture {args.fps} FPS | C clear; G snapshot; Q/Esc quit{remote}', flush=True)
        try:
            while not args.frames or count < args.frames:
                started = perf_counter()
                frame = source.read()
                debug.append(frame)
                result = perception.process(frame)
                count += 1
                timings.append(result.timings_ms)
                stage = perf_counter()
                key = -1
                if not args.headless:
                    images = draw(result, projector, args.max_dimension)
                    for window, image in zip(windows, images):
                        cv2.imshow(window, image)
                    key = cv2.waitKey(1) & 255
                    if key in (27, ord('q')):
                        break
                    if key == ord('c'):
                        perception.persistor.clear()
                    if any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
                        break
                if key in (ord('g'), ord('G')) or debug.consume_request():
                    snapshot_images = images if images is not None else draw(result, projector, args.max_dimension)
                    try:
                        print(f'Debug snapshot: {debug.save(result, c, snapshot_images, count)}', flush=True)
                    except Exception as exc:
                        print(f'Debug snapshot failed: {exc}', flush=True)
                rendering.append((perf_counter()-stage)*1000)
                loops.append((perf_counter()-started)*1000)
                if count % 10 == 0:
                    ids = [t.id for t in result.tracks if t.visible and t.confirmed]
                    print(f'{count} | core {result.timings_ms["total"]:.1f} ms | '
                          f'loop {loops[-1]:.1f} ms | dt {result.dt:.3f}s | skipped {source.skipped} | '
                          f'pair {source.pair_delta_ms:.1f} ms | IDs {ids} | {result.status}', flush=True)
        except KeyboardInterrupt:
            pass
        except Exception:
            error = traceback.format_exc()
            try:
                folder = debug.save(result, c, images, count, reason='crash', error=error)
                print(f'Crash snapshot: {folder}', flush=True)
            except Exception as snapshot_error:
                print(f'Crash snapshot failed: {snapshot_error}', flush=True)
            raise
        finally:
            if not args.headless:
                cv2.destroyAllWindows()
        if args.output_dir and result is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            images = draw(result, projector, args.max_dimension)
            for name, image in zip(('rgb','depth'), images):
                cv2.imwrite(str(args.output_dir/f'{name}.png'), image)
            details = object_details(result)
            report = dict(camera=source.name, serial=source.serial, frames=count,
                          timings_ms=list(timings), render_ms=list(rendering), loop_ms=list(loops),
                          final_status=result.status,
                          final_objects=len(details), object_details=details)
            (args.output_dir/'timings.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
