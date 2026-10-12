"""Compare local OCR engines on frame 137, or supply RGB --roi x0 y0 x1 y1.

Example: python ocr_benchmark_demo.py --engines rapidocr easyocr
EasyOCR needs CPU torch/torchvision, scipy, scikit-image, python-bidi and ninja.
"""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from rgbd_db3_reader import RealSenseReader
from rgbd_ocr import RegionOCR
from rgbd_projection import Projector
from rgbd_tabletop_segmentation import bounding_contour, segment_tabletop_objects


def object_regions(frame, calibration):
    result = segment_tabletop_objects(frame.depth, calibration.depth)
    c = calibration
    projector = Projector(
        c.depth.K, c.color.K, c.depth_to_color, frame.depth.shape, frame.color.shape[:2],
        depth_D=c.depth.D, color_D=c.color.D,
        depth_model=c.depth.distortion_model, color_model=c.color.distortion_model,
    )
    _, inverse = projector.build_maps(frame.depth)
    valid = inverse[:, :, 0] >= 0
    x, y = inverse[valid].T
    labels = np.full(frame.color.shape[:2], -1, dtype=np.int32)
    labels[valid] = result.object_labels[y, x]
    padding = round(result.bottom_padding_pixels * frame.color.shape[0]/frame.depth.shape[0])
    regions = []
    for obj in result.objects:
        pixels = cv2.findNonZero((labels == obj.label).astype(np.uint8))
        if pixels is None:  # A depth cluster can be outside/occluded in the RGB view.
            continue
        contour = bounding_contour(pixels, labels.shape, padding)
        regions.append((*map(int, contour[0]), *map(int, contour[2]+1)))
    return regions


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", default="recordings/20261005_195417.db3")
    parser.add_argument("--frame", type=int, default=137)
    parser.add_argument("--engines", nargs="+", choices=("rapidocr", "easyocr"), default=["rapidocr", "easyocr"])
    parser.add_argument("--roi", type=int, nargs=4, action="append", help="RGB crop: x0 y0 x1 y1 (exclusive)")
    parser.add_argument("--scale", type=float, default=2)
    parser.add_argument("--output", type=Path, default=Path("ocr_benchmark"))
    args = parser.parse_args()
    with RealSenseReader(args.bag, rotations=1) as reader:
        frame = reader[args.frame]
        regions = args.roi or object_regions(frame, reader.calibration)
    args.output.mkdir(parents=True, exist_ok=True)
    for index, (x0, y0, x1, y1) in enumerate(regions):
        cv2.imwrite(str(args.output / f"object_{index}.png"),
                    cv2.cvtColor(frame.color[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    report = {"bag": args.bag, "frame": args.frame, "regions": regions, "scale": args.scale, "engines": {}}
    for engine in args.engines:
        if engine == "easyocr":
            import torch
            torch.set_num_threads(4)
        started = perf_counter()
        ocr = RegionOCR(engine)
        load_ms = (perf_counter()-started)*1000
        print(f"{engine}: model initialization {load_ms:.0f} ms", flush=True)
        results = []
        for index, roi in enumerate(regions):
            readings = ocr.read_regions(frame.color, [roi], scale=args.scale)
            for reading in readings:
                reading.region = index
                print(f"  object {index}, CCW {reading.rotation:3}: {reading.elapsed_ms:.0f} ms | "
                      + " | ".join(f"{t.text!r} ({t.confidence:.2f})" for t in reading.texts), flush=True)
            results.extend(readings)
        package = "rapidocr_onnxruntime" if engine == "rapidocr" else engine
        report["engines"][engine] = {"version": version(package), "load_ms": load_ms,
                                     "results": [asdict(r) for r in results]}
        (args.output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved crops and raw results to {args.output}")


if __name__ == "__main__":
    main()
