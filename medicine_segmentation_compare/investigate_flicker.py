"""Capture/replay a short depth sequence and trace tabletop acceptance gates."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from comparison import calibration_dict
from rgbd_types import Camera
import rgbd_tabletop_segmentation as segmentation


def traced_segmenter():
    """Instrument an in-memory copy, only in this diagnostic process."""
    source = inspect.getsource(segmentation.segment_tabletop_objects)
    replacements = {
        '                if (\n                    len(inliers)':
        "                trace['planes'].append(dict(index=candidate_index, inliers=len(inliers), plane=candidate.tolist(), required=params.plane_min_inliers*sample_fraction, normal_ok=abs(float(candidate[1])) >= params.support_normal_y_min))\n                if (\n                    len(inliers)",
        '        if len(member_indices) < params.min_object_pixels:':
        "        if len(member_indices):\n            trace['clusters'].append(dict(pixels=len(member_indices), required=params.min_object_pixels, bbox=[int(pixel_x[member_indices].min()),int(pixel_y[member_indices].min()),int(pixel_x[member_indices].max()),int(pixel_y[member_indices].max())], peak=float(signed_full_distances[member_indices].max()), required_peak=params.min_object_peak_height))\n        if len(member_indices) < params.min_object_pixels:",
        '            if float(points[part_indices, 2].mean()) > params.max_object_centroid_depth:':
        "            trace['parts'].append(dict(pixels=len(part_indices), bbox=[int(pixel_x[part_indices].min()),int(pixel_y[part_indices].min()),int(pixel_x[part_indices].max()),int(pixel_y[part_indices].max())], depth=float(points[part_indices,2].mean()), extent=float(np.ptp(points[part_indices],axis=0).max())))\n            if float(points[part_indices, 2].mean()) > params.max_object_centroid_depth:",
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, f'Diagnostic hook no longer matches: {old}'
        source = source.replace(old, new)
    # Recursive half-resolution calls resolve to the instrumented function.
    exec(source, segmentation.__dict__)
    return segmentation.segment_tabletop_objects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=int, default=0)
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    folder = args.folder or Path('outputs/flicker')/datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    if args.capture:
        from rgbd_live import RealSenseLive
        with RealSenseLive() as source:
            for _ in range(30):
                source.read()
            frames = [source.read() for _ in range(args.capture)]
            (folder/'calibration.json').write_text(json.dumps(calibration_dict(source.calibration), indent=2))
        np.savez_compressed(folder/'sequence.npz', depth=np.stack([f.depth for f in frames]),
                            rgb=np.stack([f.color for f in frames]),
                            timestamp_ns=np.array([f.timestamp_ns for f in frames], np.int64))
        print(f'Captured {len(frames)} frames: {folder}', flush=True)
        return
    c = json.loads((folder/'calibration.json').read_text())['depth']
    camera = Camera(c['width'], c['height'], np.array(c['K']), np.array(c['D']), c['distortion_model'])
    data = np.load(folder/'sequence.npz')
    segment = traced_segmenter()
    base = segmentation.TabletopSegmentationParameters(max_depth=1.1, plane_min_inliers=300)
    variants = dict(current=base, lower_inliers=replace(base, plane_min_inliers=100),
                    denser_plane_samples=replace(base, plane_stride=2),
                    more_candidates=replace(base, plane_candidate_count=8),
                    fixed_first_plane=base)
    all_rows = []
    for name, parameters in variants.items():
        frozen_plane = None
        if name == 'fixed_first_plane':
            # Diagnostic control for this static sequence, not a live tracker.
            frozen_plane = np.array(next(r['plane'] for r in all_rows if r['plane']))
        for index, depth in enumerate(data['depth']):
            trace = dict(planes=[], clusters=[], parts=[])
            segmentation.trace = trace
            started = perf_counter()
            try:
                result = segment(depth, camera, parameters, processing_scale=.5, plane_model=frozen_plane)
                row = dict(error='', plane=result.plane_model,
                    objects=[dict(label=o.label, pixels=o.pixel_count,
                                  bbox=o.contour_depth_pixels.tolist(), centroid=o.centroid_xyz,
                                  peak=o.max_height_above_plane) for o in result.objects])
            except (RuntimeError, ValueError) as exc:
                result = None
                row = dict(error=str(exc), plane=None, objects=[])
            row.update(variant=name, index=index, timestamp_ns=int(data['timestamp_ns'][index]),
                       valid_fraction=float(np.mean(depth > 0)), trace=trace,
                       ms=(perf_counter()-started)*1000)
            all_rows.append(row)
            if name == 'current':
                if index == 0 or not row['objects'] or index % 10 == 0:
                    image = cv2.cvtColor(data['rgb'][index], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(folder/f'rgb_{index:03d}.png'), image)
                    view = cv2.applyColorMap(np.uint8(np.clip(depth/1.5*255, 0, 255)), cv2.COLORMAP_TURBO)
                    view[depth <= 0] = 0
                    if result is not None:
                        for obj in result.objects:
                            cv2.polylines(view, [obj.contour_depth_pixels], True, (0, 255, 0), 2)
                    cv2.imwrite(str(folder/f'depth_{index:03d}.png'), view)
        print(name, 'done', flush=True)
    (folder/'frames.json').write_text(json.dumps(all_rows, indent=2))
    from collections import Counter
    summary = {}
    for name, parameters in variants.items():
        rows = [r for r in all_rows if r['variant'] == name]
        planes = np.array([r['plane'] for r in rows if r['plane'] is not None])
        summary[name] = dict(parameters=asdict(parameters), frames=len(rows),
            errors=dict(Counter(r['error'] for r in rows if r['error'])),
            object_counts=dict(Counter(len(r['objects']) for r in rows)),
            median_ms=float(np.median([r['ms'] for r in rows])),
            plane_d_range=None if not len(planes) else [float(planes[:,3].min()), float(planes[:,3].max())])
        if name == 'fixed_first_plane':
            summary[name]['note'] = 'Diagnostic only: first successful plane reused on this static sequence; not a moving-camera implementation.'
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
