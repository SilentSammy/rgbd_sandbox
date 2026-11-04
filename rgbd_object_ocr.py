"""Background OCR attached to capture-time persistent IDs, never old pixels."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import re
from time import perf_counter
import traceback

import cv2
import numpy as np

from ocr_label_setup import LABELS, MIN_MATCH_MARGIN, MIN_MATCH_SCORE
from rgbd_label_matching import LabelMatcher, normalize
from rgbd_ocr import RegionOCR


def summarize(readings, seeds, matcher):
    items = []
    for region, seed in enumerate(seeds):
        raw = {}
        for reading in readings:
            if reading.region != region:
                continue
            for text in reading.texts:
                key = normalize(text.text)
                if key and (key not in raw or text.confidence > raw[key].confidence):
                    raw[key] = text
        texts = sorted(raw.values(), key=lambda t: t.confidence, reverse=True)
        matches = {}
        for text in texts:
            for candidate in [text.text, *re.findall(r'\w+', text.text)]:
                match = matcher.match(candidate)
                if match and match.accepted and (match.label not in matches or match.score > matches[match.label]['score']):
                    matches[match.label] = dict(asdict(match), evidence=text.text, ocr_confidence=text.confidence)
        items.append(dict(id=seed['id'], label=next(iter(matches)) if len(matches) == 1 else None,
                          ambiguous=len(matches) > 1, matches=list(matches.values()),
                          raw=[asdict(t) for t in texts], anchor=seed['anchor'].tolist(),
                          box=seed['box'], box_source=seed.get('box_source', 'depth')))
    return items


def retain_best(previous, item, timestamp_ns, elapsed_seconds):
    """Keep unambiguous accepted evidence; quality scores are not probabilities."""
    best = None if previous is None else previous.get('best_match')
    old_label = None if best is None else best['label']
    candidate = item['matches'][0] if item['label'] is not None and not item['ambiguous'] else None
    quality = lambda match: (match['score'], match['ocr_confidence'], match['margin'])
    improved = candidate is not None and (best is None or quality(candidate) > quality(best))
    if improved:
        best = dict(candidate, timestamp_ns=timestamp_ns, elapsed_seconds=elapsed_seconds)
    conflict = best is not None and (any(m['label'] != best['label'] for m in item['matches'])
                                    or (old_label is not None and old_label != best['label']))
    return dict(item, latest_label=item['label'], latest_ambiguous=item['ambiguous'],
                label=None if best is None else best['label'], best_match=best,
                label_remembered=best is not None and not improved, label_conflict=conflict,
                timestamp_ns=timestamp_ns, elapsed_seconds=elapsed_seconds)


class ObjectOCR:
    """One worker and one outstanding job; refresh timers use capture-time dt."""

    def __init__(self, output, *, refresh_seconds=5., scale=2., threads=2):
        self.output = Path(output)
        self.refresh_seconds, self.scale = refresh_seconds, scale
        self.matcher = LabelMatcher(LABELS, MIN_MATCH_SCORE, MIN_MATCH_MARGIN)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='object-ocr')
        self.future = self.executor.submit(RegionOCR, 'rapidocr', threads=threads)
        self.state, self.reader, self.error = 'loading', None, ''
        self.cache, self.last_attempt = {}, {}
        self.epoch, self.counter, self.job = 0, 0, None
        self.completed_jobs, self.last_ms = 0, None

    def clear(self):
        self.epoch += 1
        self.cache.clear()
        self.last_attempt.clear()

    def poll(self, tracks):
        current = {t.id: t for t in tracks}
        self.cache = {i: value for i, value in self.cache.items() if i in current}
        self.last_attempt = {i: value for i, value in self.last_attempt.items() if i in current}
        if self.future is None or not self.future.done():
            return None
        try:
            value = self.future.result()
            self.error = ''
            if self.state == 'loading':
                self.reader, self.state = value, 'idle'
                return dict(kind='ocr_ready')
            self.state = 'idle'
            self.completed_jobs += 1
            self.last_ms = value['elapsed_ms']
            accepted = []
            if self.job['epoch'] == self.epoch:
                for item in value['items']:
                    track = current.get(item['id'])
                    verified = (getattr(track, 'appearance_verified_at', -1.) >= self.job['elapsed_seconds']
                                and getattr(track, 'association', '') in ('appearance', 'position+appearance'))
                    if track is not None and track.confirmed and (np.linalg.norm(track.position-item['anchor']) <= .08 or verified):
                        self.cache[item['id']] = retain_best(self.cache.get(item['id']), item,
                            self.job['timestamp_ns'], self.job['elapsed_seconds'])
                        accepted.append(item['id'])
            return dict(kind='ocr_result', timestamp_ns=self.job['timestamp_ns'],
                        elapsed_ms=self.last_ms, accepted_ids=accepted, folder=value['folder'])
        except Exception:
            self.error = traceback.format_exc()
            self.state = 'failed' if self.reader is None else 'idle'
            return dict(kind='ocr_error', error=self.error)
        finally:
            self.future = None

    def submit(self, frame, seeds, elapsed_seconds, *, force=False):
        if self.state != 'idle':
            return False
        seeds = [s for s in seeds if s['id'] is not None and s['confirmed']
                 and (force or s['id'] not in self.last_attempt
                      or elapsed_seconds-self.last_attempt[s['id']] >= self.refresh_seconds)]
        if not seeds:
            return False
        self.counter += 1
        self.job = dict(epoch=self.epoch, timestamp_ns=int(frame.timestamp_ns),
                        elapsed_seconds=elapsed_seconds)
        # Own the capture image; no future display/source mutation can change it.
        rgb = frame.color.copy()
        h, w = rgb.shape[:2]
        prepared = []
        for seed in seeds:
            x0, y0, x1, y1 = seed['box']  # Inclusive segmentation bounds.
            bounds = [max(0, x0-4), max(0, y0-4), min(w, x1+5), min(h, y1+5)]
            prepared.append(dict(seed, box=bounds, anchor=seed['anchor'].copy()))
            self.last_attempt[seed['id']] = elapsed_seconds
        target = self.output/f'ocr_{self.counter:04d}_{frame.timestamp_ns}'
        self.future = self.executor.submit(self._read, rgb, prepared, dict(self.job), target)
        self.state = 'busy'
        return True

    def _read(self, rgb, seeds, job, folder):
        started = perf_counter()
        readings = self.reader.read_regions(rgb, [s['box'] for s in seeds], scale=self.scale)
        elapsed_ms = (perf_counter()-started)*1000
        items = summarize(readings, seeds, self.matcher)
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder/'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        for seed in seeds:
            x0, y0, x1, y1 = seed['box']
            cv2.imwrite(str(folder/f'id_{seed["id"]}.png'), cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
        report = dict(**job, elapsed_ms=elapsed_ms, items=items,
                      readings=[asdict(r) for r in readings], rotations=[0, 90, 180, 270])
        (folder/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return dict(items=items, elapsed_ms=elapsed_ms, folder=str(folder))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
