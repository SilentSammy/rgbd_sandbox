"""Compare original, resize-only, and batched OCR; test rotated label crops.

python ocr_speed_benchmark.py --frames 131 137 180 --repeats 3
Timing excludes model loading, segmentation, and warmup. No GPU is required.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ocr_benchmark_demo import object_regions
from ocr_label_setup import LABELS
from rgbd_db3_reader import RealSenseReader
from rgbd_label_matching import LabelMatcher
from rgbd_ocr import RegionOCR


def accepted_labels(readings, matcher):
    labels = {}
    for reading in readings:
        for text in reading.texts:
            match = matcher.match(text.text)
            if match and match.accepted:
                labels.setdefault(str(reading.region), set()).add(match.label)
    return {i: sorted(names) for i, names in labels.items()}


def rotated_regions(image, regions, degrees):
    """Rotate each original crop, preserving all its pixels on a white canvas."""
    crops = []
    for x0, y0, x1, y1 in regions:
        crop = image[y0:y1, x0:x1]
        if degrees % 90 == 0:
            crops.append(np.ascontiguousarray(np.rot90(crop, degrees//90)))
            continue
        h, w = crop.shape[:2]
        transform = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), degrees, 1)
        nw = int(np.ceil(abs(transform[0, 0])*w+abs(transform[0, 1])*h))
        nh = int(np.ceil(abs(transform[0, 1])*w+abs(transform[0, 0])*h))
        transform[:, 2] += [(nw-w)/2, (nh-h)/2]
        crops.append(cv2.warpAffine(crop, transform, (nw, nh), borderValue=(255,)*3))
    canvas = np.full((max(c.shape[0] for c in crops), sum(c.shape[1] for c in crops), 3), 255, np.uint8)
    boxes, offset = [], 0
    for crop in crops:
        h, w = crop.shape[:2]
        canvas[:h, offset:offset+w] = crop
        boxes.append((offset, 0, offset+w, h))
        offset += w
    return canvas, boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", default="recordings/20261005_195417.db3")
    parser.add_argument("--frames", nargs="+", type=int, default=[131, 137, 180])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--angles", nargs="*", type=int, default=list(range(0, 360, 45)),
                        help="crop rotation tests; pass --angles alone to skip")
    parser.add_argument("--output", type=Path, default=Path("ocr_benchmark/speed_results.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    with RealSenseReader(args.bag, rotations=1) as reader:
        samples = [(i, reader[i]) for i in args.frames]
        samples = [(i, f.color, object_regions(f, reader.calibration)) for i, f in samples]
    matcher = LabelMatcher(LABELS)
    report = {"bag": args.bag, "repeats": args.repeats, "frames": {}, "orientation_tests": {}}
    for mode in ("original", "resize_only", "batched"):
        ocr = RegionOCR(fast=mode != "original")
        if mode == "resize_only":
            ocr.fast = False
        for index, image, regions in samples:
            ocr.read_regions(image, regions)  # Warm each image shape before timing.
            times, readings = [], []
            for _ in range(args.repeats):
                started = perf_counter()
                readings = ocr.read_regions(image, regions)
                times.append((perf_counter()-started)*1000)
            result = {"times_ms": times, "median_ms": float(np.median(times)),
                      "accepted_labels": accepted_labels(readings, matcher),
                      "readings": [asdict(r) for r in readings]}
            report["frames"].setdefault(str(index), {"regions": regions})[mode] = result
            print(f"{mode}, frame {index}: {result['median_ms']:.0f} ms | {result['accepted_labels']}", flush=True)
        # Test the first requested frame at eight angles, including upside-down
        # and diagonal text. Compare both pipelines on precisely the same crops.
        if mode in ("original", "batched"):
            index, image, regions = samples[0]
            for degrees in args.angles:
                rotated, boxes = rotated_regions(image, regions, degrees)
                started = perf_counter()
                readings = ocr.read_regions(rotated, boxes)
                result = {"frame": index, "elapsed_ms": (perf_counter()-started)*1000,
                          "accepted_labels": accepted_labels(readings, matcher)}
                report["orientation_tests"].setdefault(str(degrees), {})[mode] = result
                print(f"{mode}, {degrees} degrees: {result['elapsed_ms']:.0f} ms | {result['accepted_labels']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
