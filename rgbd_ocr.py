"""Local OCR of selected RGB regions; boxes use exclusive (x0, y0, x1, y1).

Reuse one RegionOCR instance to avoid loading models for every image.
Returned text polygons are in the original, full-resolution image coordinates.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


@dataclass
class OCRText:
    text: str
    confidence: float
    polygon: list[list[float]]


@dataclass
class OCRResult:
    region: int
    rotation: int  # Crop/direction hypothesis in CCW degrees; not a measured text angle.
    elapsed_ms: float
    texts: list[OCRText]


class RegionOCR:
    def __init__(self, engine="rapidocr", model_directory=None, *, fast=True):
        self.engine = engine
        self.fast = fast and engine == "rapidocr"
        if engine == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR
            options = dict(det_limit_type="max") if fast else {}
            self.reader = RapidOCR(text_score=0.3, intra_op_num_threads=4, inter_op_num_threads=1,
                                   **options)
        elif engine == "easyocr":
            import easyocr
            directory = Path(model_directory or Path(__file__).parent / ".ocr_models" / "easyocr")
            self.reader = easyocr.Reader(
                ["en", "es"], gpu=False, model_storage_directory=str(directory),
                user_network_directory=str(directory / "user_network"), verbose=False,
            )
        else:
            raise ValueError("engine must be rapidocr or easyocr")

    def read_regions(self, image_rgb, regions, *, rotations=(0, 1, 2, 3), scale=2, min_confidence=0.3):
        """Test quarter-turn hypotheses, retaining results (including disagreements).

        Enlargement helps the recognizer but does not restore lost image detail.
        Confidence scores are engine-specific, not probabilities of correctness.
        """
        if scale <= 0:
            raise ValueError("scale must be positive")
        regions, rotations = tuple(regions), tuple(rotations)
        if any(k not in range(4) for k in rotations):
            raise ValueError("rotations must contain quarter-turns 0..3")
        for x0, y0, x1, y1 in regions:
            if not (0 <= x0 < x1 <= image_rgb.shape[1] and 0 <= y0 < y1 <= image_rgb.shape[0]):
                raise ValueError("region must lie inside the image")
        if getattr(self, "fast", False):
            return self._read_fast(image_rgb, regions, rotations, scale, min_confidence)
        results = []
        for region, (x0, y0, x1, y1) in enumerate(regions):
            crop = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
            height, width = crop.shape[:2]
            for k in rotations:
                rotated = np.ascontiguousarray(np.rot90(crop, k))
                enlarged = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                started = perf_counter()
                if self.engine == "rapidocr":
                    found, _ = self.reader(enlarged)
                    found = found or []
                else:
                    found = self.reader.readtext(enlarged)
                elapsed = (perf_counter()-started)*1000
                texts = []
                for polygon, text, confidence in found:
                    if confidence < min_confidence:
                        continue
                    points = np.asarray(polygon, dtype=float)
                    points = (points+0.5) / np.array(
                        [enlarged.shape[1]/rotated.shape[1], enlarged.shape[0]/rotated.shape[0]]
                    ) - 0.5
                    x, y = points.T
                    if k == 1:
                        points = np.column_stack((width-1-y, x))
                    elif k == 2:
                        points = np.column_stack((width-1-x, height-1-y))
                    elif k == 3:
                        points = np.column_stack((y, height-1-x))
                    points += (x0, y0)
                    texts.append(OCRText(text, float(confidence), points.tolist()))
                results.append(OCRResult(region, k*90, elapsed, texts))
        return results

    def _read_fast(self, image_rgb, regions, rotations, scale, min_confidence):
        """Detect each text axis once; batch both reading directions of its lines.

        Perspective rectification handles tilted/vertical lines. Testing both
        rectified directions avoids relying on the 0/180 direction classifier.
        Model component APIs are from the pinned RapidOCR 1.x package.
        """
        results, lines, targets = [], [], []
        for region, (x0, y0, x1, y1) in enumerate(regions):
            crop = cv2.cvtColor(image_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
            height, width = crop.shape[:2]
            readings = {k: OCRResult(region, k*90, 0, []) for k in rotations}
            results.extend(readings.values())
            done = set()
            for k in rotations:
                if k % 2 in done:
                    continue
                done.add(k % 2)
                rotated = np.ascontiguousarray(np.rot90(crop, k))
                enlarged = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                started = perf_counter()
                boxes, _ = self.reader.text_det(enlarged)
                directions = [r for r in rotations if r % 2 == k % 2]
                elapsed = (perf_counter()-started)*1000 / len(directions)
                for r in directions:
                    readings[r].elapsed_ms += elapsed
                if boxes is None or not len(boxes):
                    continue
                line_images = self.reader.get_crop_img_list(enlarged, boxes)
                for polygon, line in zip(boxes, line_images):
                    points = (polygon.astype(float)+0.5) / np.array(
                        [enlarged.shape[1]/rotated.shape[1], enlarged.shape[0]/rotated.shape[0]]
                    ) - 0.5
                    x, y = points.T
                    if k == 1:
                        points = np.column_stack((width-1-y, x))
                    elif k == 2:
                        points = np.column_stack((width-1-x, height-1-y))
                    elif k == 3:
                        points = np.column_stack((y, height-1-x))
                    points += (x0, y0)
                    for r in directions:
                        lines.append(line if r == k else np.ascontiguousarray(np.rot90(line, 2)))
                        targets.append((readings[r], points.tolist()))
        if lines:
            started = perf_counter()
            recognized, _ = self.reader.text_rec(lines)
            elapsed = (perf_counter()-started)*1000 / len(lines)
            for (reading, polygon), (text, confidence) in zip(targets, recognized):
                reading.elapsed_ms += elapsed
                if confidence >= min_confidence:
                    reading.texts.append(OCRText(text, float(confidence), polygon))
        return results
