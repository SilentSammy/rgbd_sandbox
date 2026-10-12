"""Export a low-frame-rate MP4 with tabletop rectangles and local OCR.

python export_rgbd_ocr.py --fps 2
No tracking or inferred labels: every sampled frame is independently processed.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unicodedata
from time import perf_counter

import cv2
import numpy as np

from ocr_benchmark_demo import object_regions
from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import _resize
from rgbd_ocr import RegionOCR
from tabletop_segmentation_demo import OBJECT_COLORS


def annotate(frame, regions, readings, index, elapsed, error="", matcher=None):
    image = cv2.cvtColor(frame.color, cv2.COLOR_RGB2BGR)
    for region, (x0, y0, x1, y1) in enumerate(regions):
        color = tuple(map(int, OBJECT_COLORS[region % len(OBJECT_COLORS)]))
        cv2.rectangle(image, (x0, y0), (x1-1, y1-1), color, 3)
        cv2.putText(image, str(region+1), (x0, max(y0-8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    image = _resize(image, 960)
    panel = np.full((image.shape[0], 420, 3), 25, dtype=np.uint8)

    def text(value, y, color=(230, 230, 230), scale=0.55):
        cv2.putText(panel, value, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    text("RGB-D segmentation + OCR", 35, scale=0.7)
    text(f"Source frame {index} | {elapsed:.2f} s", 68)
    text("Tabletop geometry / RapidOCR", 98)
    text("Independent frame processing", 128)
    for region in range(len(regions)):
        color = tuple(map(int, OBJECT_COLORS[region % len(OBJECT_COLORS)]))
        y = 180 + region*min(140, 640//max(len(regions), 1))
        text(f"Region {region+1}", y, color, 0.65)
        found = {}
        for reading in readings:
            if reading.region != region:
                continue
            for item in reading.texts:
                key = item.text.casefold().strip(" .'\"")
                if len(key) >= 3 and (key not in found or item.confidence > found[key].confidence):
                    found[key] = item
        def rank(item):
            match = matcher.match(item.text) if matcher else None
            return (bool(match and match.accepted), match.score if match else 0, item.confidence)

        words = sorted(found.values(), key=rank if matcher else lambda t: t.confidence, reverse=True)[:2 if matcher else 3]
        if not words:
            text("No confident text", y+30, (170, 170, 170))
        for line, item in enumerate(words):
            # OpenCV's built-in font supports ASCII; exact Unicode stays in JSON.
            label = item.text.encode("ascii", "replace").decode("ascii")[:33]
            text(f"{label}  [{item.confidence:.2f}]", y+30+line*(50 if matcher else 25))
            if matcher:
                match = matcher.match(item.text)
                suggested = "Unknown label"
                if match and match.score >= 0.5:
                    name = unicodedata.normalize("NFKD", match.label).encode("ascii", "ignore").decode("ascii")
                    prefix = "->" if match.accepted else "Uncertain:"
                    suggested = f"{prefix} {name} [match {match.score:.2f}]"
                text(suggested, y+53+line*50, color if match and match.accepted else (150, 180, 230), 0.48)
    if error:
        text("Support-plane detection failed", 180, (100, 100, 255))
    text("Rectangles include bottom padding", 900, scale=0.48)
    text("Dictionary suggestions, not product IDs" if matcher else
         "Scores are not certainty of correctness", 925, scale=0.48)
    return np.hstack((image, panel))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", default="recordings/20261005_195417.db3")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--output", type=Path, default=Path("exports/20261005_195417_segmentation_ocr.mp4"))
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("fps must be finite and positive")
    if args.output.exists():
        raise FileExistsError(f"Choose a new output path: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ocr = RegionOCR("rapidocr")
    records = []
    started = perf_counter()
    writer = None
    try:
        with RealSenseReader(args.bag, rotations=1) as reader:
            first = reader[0].timestamp_ns
            duration = (reader[len(reader)-1].timestamp_ns-first)/1e9
            count = min(len(reader), max(1, round(duration*args.fps)))
            indices = np.rint(np.linspace(0, len(reader)-1, count)).astype(int)
            # Include the reference frame when exporting the benchmark recording.
            if Path(args.bag).name == "20261005_195417.db3" and len(reader) > 137 and count > 2:
                indices[np.argmin(abs(indices-137))] = 137
                indices.sort()
            fps = count/duration if duration > 0 else args.fps
            for progress, index in enumerate(indices, 1):
                frame = reader[int(index)]
                elapsed = (frame.timestamp_ns-first)/1e9
                error = ""
                try:
                    regions = object_regions(frame, reader.calibration)
                    regions = [r for r in regions if r[0] < r[2] and r[1] < r[3]]
                except RuntimeError as exc:
                    regions, error = [], str(exc)
                readings = ocr.read_regions(frame.color, regions, min_confidence=0.8)
                annotated = annotate(frame, regions, readings, int(index), elapsed, error)
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                            fps, (annotated.shape[1], annotated.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Could not open MP4 encoder")
                writer.write(annotated)
                if index == 137:
                    cv2.imwrite(str(args.output.with_suffix(".png")), annotated)
                records.append({"frame": int(index), "time_s": elapsed, "regions": regions,
                                "ocr": [asdict(r) for r in readings], "error": error})
                print(f"{progress}/{count}: frame {index}, {len(regions)} regions | "
                      f"{perf_counter()-started:.1f} s processing elapsed", flush=True)
    finally:
        if writer is not None:
            writer.release()
    report = {"bag": args.bag, "video_fps": fps, "duration_s": duration,
              "rotations_ccw": [0, 90, 180, 270], "confidence_threshold": 0.8, "frames": records}
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}: {count} frames, {fps:.3f} fps, {duration:.2f} s")


if __name__ == "__main__":
    main()
