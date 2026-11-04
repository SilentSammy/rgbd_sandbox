"""Medicine mask experiment: SAM 2.1 versus the current depth baseline."""

import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter
import traceback

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import cv2
import numpy as np

from comparison import Comparison, SnapshotSource, RGBD_ROOT, box, export, panels, proposals
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _native_pixel, _resize
from sam_backend import SamSegmenter


def load_source(args):
    if args.live:
        from rgbd_live import RealSenseLive
        return RealSenseLive(serial=args.serial,rotations=args.rotations)
    if args.source.is_dir():
        return SnapshotSource(args.source)
    from rgbd_db3_reader import RealSenseReader
    return RealSenseReader(args.source,rotations=args.rotations)


def prompt_points(value):
    points = []
    for pair in value.split(';'):
        x,y = map(float,pair.split(','))
        points.append(dict(kind='point',coordinates=[x,y],source='manual_point'))
    return points


def run(args):
    cv2.setNumThreads(1)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = args.output_dir/stamp
    run_dir.mkdir(parents=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['rgbd_module_sha256'] = {}
    import hashlib
    for name in ('rgbd_tabletop_segmentation.py','rgbd_projection.py','rgbd_db3_reader.py'):
        config['rgbd_module_sha256'][name] = hashlib.sha256((RGBD_ROOT/name).read_bytes()).hexdigest()
    (run_dir/'config.json').write_text(json.dumps(config,indent=2))
    print(f'Outputs: {run_dir}',flush=True)
    try:
        sam = None if args.baseline_only else SamSegmenter(args.model,args.device,args.revision)
        versions = {}
        for name in ('numpy','opencv-python','torch','transformers','open3d'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        model_info = dict(engine='depth_only' if sam is None else 'sam2_image',
            model=None if sam is None else sam.model_id,
            model_revision=None if sam is None else sam.revision,
            device=None if sam is None else sam.device,
            model_load_ms=None if sam is None else sam.load_ms,
            model_loading_info=None if sam is None else sam.loading_info,
            versions=versions)
        (run_dir/'environment.json').write_text(json.dumps(model_info,indent=2))
        source = load_source(args)
        with closing(source):
            comparator = Comparison(source.calibration,max_depth=args.max_object_depth,scale=args.segmentation_scale)
            if args.live:
                playback = None
                index = 0
            else:
                start = args.start_frame if args.start_frame is not None else (0 if args.source.is_dir() else min(137,len(source)-1))
                playback = Playback(source,start)
                index = start
            controls = dict(clicks=prompt_points(args.points) if args.points else [],dirty=True,frame=None,display=None)
            def mouse(event,x,y,flags,param):
                if event == cv2.EVENT_LBUTTONDOWN and controls['frame'] is not None:
                    point = _native_pixel(x,y,controls['display'],controls['frame'].color.shape)
                    controls['clicks'].append(dict(kind='point',coordinates=list(point),source='manual_click'))
                    controls['dirty'] = True
            windows = ('Depth baseline','SAM medicine experiment')
            if not args.headless:
                for name in windows:
                    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(windows[1],mouse)
            print(PLAYBACK_HELP+' | click SAM: prompt | C clear prompts | B depth prompts | G export',flush=True)
            displayed, processed, visit, previous_masks = None,0,0,None
            previous_index = None
            gallery = []
            last_export = None
            while True:
                if args.live or index != displayed or controls['dirty']:
                    started = perf_counter()
                    frame = source.read() if args.live else source[index]
                    baseline = comparator.baseline(frame)
                    current_prompts = controls['clicks'] or proposals(baseline['labels'])
                    if args.track_boxes and not controls['dirty'] and previous_masks is not None and (args.live or index == previous_index+1):
                        current_prompts = [dict(kind='box',coordinates=bounds,source='previous_sam_box',slot=i)
                            for i,mask in enumerate(previous_masks) if (bounds:=box(mask)) is not None]
                    for prompt in current_prompts:
                        x,y = prompt['coordinates'][:2]
                        if not (0 <= x < frame.color.shape[1] and 0 <= y < frame.color.shape[0]):
                            raise ValueError(f'Prompt outside RGB image: {prompt}')
                    masks = np.zeros((0,*frame.color.shape[:2]),bool)
                    scores, sam_timings = [],dict(preprocess_ms=0,inference_ms=0,postprocess_ms=0)
                    if sam is not None:
                        masks,scores,sam_timings = sam.segment(frame.color,current_prompts)
                    images = panels(frame,baseline,masks,sam_ran=sam is not None)
                    inference_total_ms = (perf_counter()-started)*1000
                    metadata = dict(**model_info,index=index,visit=visit,source=str(args.source) if not args.live else 'live',
                        prompt_mode='manual' if controls['clicks'] else ('tracked_boxes' if args.track_boxes else 'depth_proposals'),
                        prompts=current_prompts,baseline_parameters=asdict(comparator.parameters),
                        sam_timings_ms=sam_timings,processing_and_render_ms=inference_total_ms,
                        persistent_ids=False)
                    frame_folder = f'frame_{index:06d}_visit_{visit:06d}'
                    export_started = perf_counter()
                    if last_export is None or export_started-last_export >= args.save_interval or controls['dirty']:
                        export(run_dir/frame_folder,frame,baseline,masks,
                               current_prompts,scores,metadata,source.calibration)
                        last_export = export_started
                        gallery.append(f'<p>Frame {index}, visit {visit}: <a href="{frame_folder}/report.json">measurements</a></p>'
                                       f'<a href="{frame_folder}/comparison.png"><img style="max-width:100%" src="{frame_folder}/comparison.png"></a>')
                        (run_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Medicine segmentation comparison</title>'
                            '<h1>Depth baseline / prompted SAM</h1><p>Mask agreement is not ground-truth accuracy. Numbers are per-frame mask indices, not persistent IDs.</p>'
                            +''.join(gallery),encoding='utf-8')
                    with (run_dir/'summary.jsonl').open('a') as file:
                        file.write(json.dumps(dict(index=index,visit=visit,baseline_ms=baseline['segmentation_ms'],
                            sam_ms=sum(sam_timings.values()),export_ms=(perf_counter()-export_started)*1000,
                            objects=len(masks),baseline_error=baseline['error']))+'\n')
                    print(f'Frame {index}: depth {baseline["segmentation_ms"]:.1f} ms; SAM {sum(sam_timings.values()):.1f} ms; '
                          f'{len(current_prompts)} prompts -> {len(masks)} masks; {baseline["error"] or "plane OK"}',flush=True)
                    previous_index,previous_masks = index,masks
                    controls.update(frame=frame,dirty=False)
                    if not args.headless:
                        for name,img in zip(windows,images):
                            shown = _resize(img,args.max_dimension)
                            if name == windows[1]:
                                controls['display'] = shown.shape
                            cv2.imshow(name,shown)
                        if processed == 0:
                            cv2.moveWindow(windows[0],20,40)
                            cv2.moveWindow(windows[1],40+shown.shape[1],40)
                    processed += 1
                    visit += 1
                    displayed = index
                    if args.frames and processed >= args.frames:
                        break
                    if args.headless:
                        index += args.stride
                        if not args.live and index >= len(source):
                            break
                        continue
                key = cv2.waitKey(1 if args.live or (playback and playback.direction) else 20)
                if playback:
                    command = playback.update(key)
                    index = playback.index
                    if playback.quit:
                        break
                else:
                    command = chr(key & 255).lower() if key >= 0 else ''
                    if command == '\x1b':
                        break
                    index += 1
                if command in ('c','b'):
                    controls['clicks'].clear()
                    previous_masks = None
                    controls['dirty'] = True
                if command == 'g':
                    export(run_dir/f'manual_{datetime.now().strftime("%H%M%S_%f")}',frame,baseline,masks,
                           current_prompts,scores,metadata,source.calibration)
                if any(cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1 for name in windows):
                    break
    except Exception:
        (run_dir/'traceback.txt').write_text(traceback.format_exc())
        if 'frame' in locals():
            from comparison import calibration_dict
            failure = run_dir/'failure_input'
            failure.mkdir(exist_ok=True)
            cv2.imwrite(str(failure/'raw_rgb.png'),cv2.cvtColor(frame.color,cv2.COLOR_RGB2BGR))
            np.save(failure/'depth_meters.npy',frame.depth)
            (failure/'report.json').write_text(json.dumps(dict(timestamp_ns=int(frame.timestamp_ns),
                calibration=calibration_dict(source.calibration),index=index,
                prompts=locals().get('current_prompts',[])),indent=2))
        raise
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,nargs='?',default=RGBD_ROOT/'recordings/20261005_195417.db3')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--serial')
    parser.add_argument('--start-frame',type=int)
    parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--rotations',type=int,choices=range(4),default=1)
    parser.add_argument('--max-dimension',type=int,default=600)
    parser.add_argument('--max-object-depth',type=float,default=1.05)
    parser.add_argument('--segmentation-scale',type=float,choices=(.5,1.),default=.5)
    parser.add_argument('--model',default='facebook/sam2.1-hiera-tiny')
    parser.add_argument('--revision',help='Hub commit/tag; default Tiny checkpoint is pinned to the verified revision')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--baseline-only',action='store_true',help='explicit depth-only diagnostic, no SAM model')
    parser.add_argument('--points',help='RGB point prompts: "x,y;x,y"; override depth proposals')
    parser.add_argument('--track-boxes',action='store_true',help='re-prompt with previous SAM boxes on forward steps; not SAM video-memory tracking')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--save-interval',type=float,default=0,help='minimum seconds between automatic exports; 0 saves every frame; G always saves')
    args = parser.parse_args()
    if args.frames<0 or args.stride<1 or args.max_dimension<=0 or args.save_interval<0:
        parser.error('frame count must be nonnegative; stride/dimension must be positive')
    if args.headless and not args.frames:
        args.frames = 1
    run(args)


if __name__ == '__main__':
    main()
