"""Interactive tabletop segmentation and OCR with shared viewer playback.

python segmentation_ocr_demo.py --start-frame 137
OCR runs in the background; outdated results never replace the current frame.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import cv2

from export_rgbd_ocr import annotate
from ocr_benchmark_demo import object_regions
from rgbd_db3_reader import RealSenseReader
from rgbd_db3_viewer import Playback, PLAYBACK_HELP, _resize
from rgbd_ocr import RegionOCR


def process_frame(ocr, frame, calibration, index, elapsed, label_matcher=None):
    started = perf_counter()
    error = ""
    try:
        regions = object_regions(frame, calibration)
    except RuntimeError as exc:
        regions, error = [], str(exc)
    readings = ocr.read_regions(frame.color, regions,
                                min_confidence=0.3 if label_matcher else 0.8)
    image = annotate(frame, regions, readings, index, elapsed, error, matcher=label_matcher)
    print(f"Frame {index}: {len(regions)} regions, {(perf_counter()-started)*1000:.0f} ms", flush=True)
    return index, image


def view(reader, *, start_frame=0, max_dimension=960, label_matcher=None):
    if not 0 <= start_frame < len(reader):
        raise ValueError(f"start-frame must be between 0 and {len(reader)-1}")
    if max_dimension <= 0:
        raise ValueError("max-dimension must be positive")
    ocr = RegionOCR("rapidocr")
    first_timestamp = reader[0].timestamp_ns
    playback = Playback(reader, start_frame)
    index, displayed = playback.index, None
    cache, future = {}, None
    window = "Segmentation + OCR"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    print(PLAYBACK_HELP)
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            while True:
                if index != displayed:
                    frame = reader[index]
                    elapsed = (frame.timestamp_ns-first_timestamp)/1e9
                    image = cache.get(index)
                    if image is None:
                        image = annotate(frame, [], [], index, elapsed)
                        cv2.putText(image, "Processing segmentation and OCR...", (558, 165),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
                    cv2.imshow(window, _resize(image, max_dimension))
                    displayed = index
                if future is not None and future.done():
                    completed, image = future.result()
                    cache[completed] = image
                    if len(cache) > 8:
                        cache.pop(next(iter(cache)))
                    if completed == index:
                        cv2.imshow(window, _resize(image, max_dimension))
                    future = None
                if future is None and index not in cache:
                    future = worker.submit(process_frame, ocr, frame, reader.calibration, index, elapsed, label_matcher)
                playback.poll()
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                if playback.quit:
                    break
                index = playback.index
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog=PLAYBACK_HELP)
    parser.add_argument("bag", nargs="?", default="recordings/20261005_195417.db3")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-dimension", type=int, default=960)
    args = parser.parse_args()
    with RealSenseReader(args.bag, rotations=1) as reader:
        view(reader, start_frame=args.start_frame, max_dimension=args.max_dimension)
