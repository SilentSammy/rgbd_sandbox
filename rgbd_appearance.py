"""Small masked RGB signatures for object association, not product recognition."""

from dataclasses import dataclass

import cv2
import numpy as np

from rgbd_projection import Projector


@dataclass(frozen=True)
class Signature:
    histogram: np.ndarray
    descriptors: np.ndarray | None
    points: np.ndarray
    shape: tuple


@dataclass(frozen=True)
class Observation:
    signature: Signature | None
    extent: tuple


@dataclass(frozen=True)
class Similarity:
    score: float
    color: float
    reliable: bool
    inliers: int = 0


def describe(rgb, mask):
    """Keep descriptor patches, not just keypoint centers, inside the object."""
    y, x = np.where(mask)
    if len(x) < 40:
        return None
    x0, x1 = max(0, int(x.min())-2), min(rgb.shape[1], int(x.max())+3)
    y0, y1 = max(0, int(y.min())-2), min(rgb.shape[0], int(y.max())+3)
    crop, mask = rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(np.uint8)*255
    scale = 240/max(crop.shape[:2])
    size = max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale))
    crop = cv2.resize(crop, size)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    interior = cv2.erode(mask, np.ones((17, 17), np.uint8))
    if np.count_nonzero(interior) < 40:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], interior, [12, 4], [0, 180, 0, 256])
    histogram /= max(float(histogram.sum()), 1)
    orb = cv2.ORB_create(nfeatures=180, nlevels=4, edgeThreshold=8, patchSize=15, fastThreshold=10)
    keypoints, descriptors = orb.detectAndCompute(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), interior)
    points = np.float32([p.pt for p in keypoints]).reshape(-1, 2)
    for array in (histogram, descriptors, points):
        if array is not None:
            array.setflags(write=False)
    return Signature(histogram, descriptors, points, crop.shape[:2])


def compare(first, second):
    color = float(np.clip(1-cv2.compareHist(first.histogram, second.histogram, cv2.HISTCMP_BHATTACHARYYA), 0, 1))
    # Color-only recovery is allowed only with a very high score AND a clear
    # lead over competing tracks/detections, enforced by the memory matcher.
    fallback = .8*color
    if color < .15 or first.descriptors is None or second.descriptors is None:
        return Similarity(fallback, color, False)
    if min(len(first.descriptors), len(second.descriptors)) < 6:
        return Similarity(fallback, color, False)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(first.descriptors, second.descriptors, k=2)
    good = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .75*b.distance and a.distance < 60]
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < 6:
        return Similarity(fallback, color, False)
    a = np.float32([first.points[m.queryIdx] for m in good])
    b = np.float32([second.points[m.trainIdx] for m in good])
    transform, inliers = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3,
                                            maxIters=300, confidence=.95, refineIters=5)
    if transform is None:
        return Similarity(fallback, color, False)
    scales = np.linalg.svd(transform[:, :2], compute_uv=False)
    if np.linalg.det(transform[:, :2]) <= 0 or scales.min() < .25 or scales.max() > 4:
        return Similarity(fallback, color, False)
    keep = inliers.ravel().astype(bool)
    count, ratio = int(keep.sum()), float(keep.mean())
    coverage = min(float(np.prod(np.ptp(a[keep], axis=0)))/np.prod(first.shape),
                   float(np.prod(np.ptp(b[keep], axis=0)))/np.prod(second.shape)) if count else 0
    required = max(6, min(12, round(.15*min(len(first.descriptors), len(second.descriptors)))))
    reliable = count >= required and ratio >= .65 and coverage >= .08 and color >= .15
    feature = min(1., count/12)*ratio if reliable else 0.
    score = max(fallback, .85+.15*(.75*feature+.25*color)) if reliable else fallback
    return Similarity(score, color, reliable, count)


class RGBObjectAppearance:
    """Calibrated depth masks -> RGB silhouettes -> bounded-size signatures."""

    def __init__(self, calibration):
        c = calibration
        shape = c.depth.height//2, c.depth.width//2
        K = c.depth.K.copy()
        for row, factor in enumerate((shape[1]/c.depth.width, shape[0]/c.depth.height)):
            K[row] *= factor
            K[row, 2] += (factor-1)*.5
        self.projector = Projector(K, c.color.K, c.depth_to_color, shape,
            (c.color.height, c.color.width), depth_D=c.depth.D, color_D=c.color.D,
            depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)

    def observe(self, frame, segmentation):
        size = self.projector.depth_shape[::-1]
        depth = cv2.resize(frame.depth, size, interpolation=cv2.INTER_NEAREST_EXACT)
        labels = cv2.resize(segmentation.object_labels, size, interpolation=cv2.INTER_NEAREST_EXACT)
        _, inverse = self.projector.build_maps(depth)
        valid = inverse[:, :, 0] >= 0
        rgb_labels = np.full(frame.color.shape[:2], -1, np.int32)
        x, y = inverse[valid].T
        rgb_labels[valid] = labels[y, x]
        observations = []
        for obj in segmentation.objects:
            y, x = np.where(rgb_labels == obj.label)
            if len(x) < 40:
                observations.append(Observation(None, tuple(sorted(obj.extent_xyz))))
                continue
            # Morphology and feature extraction need only this small ROI.
            x0, x1 = max(0, int(x.min())-4), min(rgb_labels.shape[1], int(x.max())+5)
            y0, y1 = max(0, int(y.min())-4), min(rgb_labels.shape[0], int(y.max())+5)
            mask = (rgb_labels[y0:y1, x0:x1] == obj.label).astype(np.uint8)
            # Fill the sampling grid's small holes without using the enclosing
            # rectangle as a foreground mask. Erosion happens in describe().
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            y, x = np.where(mask)
            if len(x) >= 40:
                # Boxes/cylinders are convex: depth holes in a printed face
                # should not exclude its RGB lettering from the signature.
                hull = cv2.convexHull(np.column_stack((x, y)).astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 1)
            observations.append(Observation(describe(frame.color[y0:y1, x0:x1], mask), tuple(sorted(obj.extent_xyz))))
        return observations
