"""Read-only bridge to our RGB-D geometry plus inspectable comparison exports."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

RGBD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RGBD_ROOT))
from rgbd_types import Camera, Calibration, Frame
from rgbd_projection import Projector
from rgbd_tabletop_segmentation import TabletopSegmentationParameters, segment_tabletop_objects


def calibration_dict(c):
    def camera(cam):
        return dict(width=cam.width, height=cam.height, K=cam.K.tolist(), D=cam.D.tolist(), distortion_model=cam.distortion_model)
    return dict(color=camera(c.color), depth=camera(c.depth), depth_to_color=c.depth_to_color.tolist())


class SnapshotSource:
    """Live debug bundle; history is already rotated and depth already meters."""
    def __init__(self, path):
        self.path = Path(path)
        self.report = json.loads((self.path/'report.json').read_text())
        c = self.report['calibration']
        def camera(values):
            return Camera(values['width'], values['height'], np.array(values['K']), np.array(values['D']), values['distortion_model'])
        self.calibration = Calibration(camera(c['color']), camera(c['depth']), np.array(c['depth_to_color']))
        self.history = np.load(self.path/'history.npz') if (self.path/'history.npz').exists() else None
        self.fps = None

    def __len__(self):
        return len(self.history['timestamps_ns']) if self.history is not None else 1

    def __getitem__(self, index):
        if self.history is not None:
            return Frame(int(self.history['timestamps_ns'][index]), self.history['color'][index].copy(), self.history['depth'][index].copy())
        if index != 0:
            raise IndexError(index)
        filename = self.path/('raw_rgb.png' if (self.path/'raw_rgb.png').exists() else 'rgb.png')
        rgb = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
        return Frame(int(self.report['timestamp_ns']), rgb, np.load(self.path/'depth_meters.npy'))

    def close(self):
        if self.history is not None:
            self.history.close()


class Comparison:
    def __init__(self, calibration, *, max_depth=1.05, scale=.5):
        self.calibration, self.scale = calibration, scale
        self.parameters = TabletopSegmentationParameters(max_object_centroid_depth=max_depth)
        c = calibration
        self.projector = Projector(c.depth.K, c.color.K, c.depth_to_color,
            (c.depth.height,c.depth.width), (c.color.height,c.color.width),
            depth_D=c.depth.D, color_D=c.color.D,
            depth_model=c.depth.distortion_model, color_model=c.color.distortion_model)

    def baseline(self, frame):
        started = perf_counter()
        error, result = None, None
        try:
            result = segment_tabletop_objects(frame.depth, self.calibration.depth, self.parameters, processing_scale=self.scale)
        except (RuntimeError, ValueError) as exc:
            error = str(exc)
        segmentation_ms = (perf_counter()-started)*1000
        started = perf_counter()
        forward, inverse = self.projector.build_maps(frame.depth)
        valid = inverse[:,:,0] >= 0
        labels = np.full(frame.color.shape[:2], -1, np.int32)
        if result is not None:
            x,y = inverse[valid].T
            labels[valid] = result.object_labels[y,x]
        return dict(result=result, labels=labels, forward=forward, inverse=inverse, error=error,
                    rays=self.projector.rays,translation=self.projector.translation,
                    segmentation_ms=segmentation_ms, projection_ms=(perf_counter()-started)*1000)


def box(mask):
    ys,xs = np.where(mask)
    return None if not len(xs) else [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())]


def proposals(labels):
    return [dict(kind='box', coordinates=box(labels==label), source='depth_proposal', baseline_label=int(label))
            for label in np.unique(labels) if label >= 0 and np.count_nonzero(labels==label) >= 10]


def overlap(first, second):
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second)/union) if union else None


def overlay(rgb, masks, title):
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for index,mask in enumerate(masks):
        tint = np.random.default_rng(index+50).integers(50,256,3)
        image[mask] = (image[mask]*.55+tint*.45).astype(np.uint8)
        contours,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, tuple(map(int,tint)), 1)
        bounds = box(mask)
        if bounds:
            cv2.putText(image,str(index),(bounds[0],max(40,bounds[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,tuple(map(int,tint)),2)
    cv2.rectangle(image,(0,0),(image.shape[1],28),(0,0,0),-1)
    cv2.putText(image,title,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    return image


def panels(frame, baseline, masks, *, sam_ran=True):
    labels = baseline['labels']
    depth_masks = [labels==label for label in np.unique(labels) if label>=0]
    left = overlay(frame.color, depth_masks, 'Depth baseline: '+(baseline['error'] or f'{len(depth_masks)} proposals'))
    right = overlay(frame.color, masks, f'SAM: {len(masks)} prompted masks' if sam_ran else 'Baseline only: SAM not run')
    return left,right


def measurements(frame, baseline, masks, prompts, scores):
    items = []
    inverse = baseline['inverse']
    valid = inverse[:,:,0] >= 0
    for i,mask in enumerate(masks):
        depth_members = np.zeros(frame.depth.shape,bool)
        xy = inverse[valid & mask]
        depth_members[xy[:,1],xy[:,0]] = True
        depths = frame.depth[depth_members]
        xyz = np.column_stack([ray[depth_members]*depths+translation
            for ray,translation in zip(baseline['rays'],baseline['translation'])])
        target = prompts[i].get('baseline_label')
        reference = baseline['labels']==target if target is not None else np.zeros(mask.shape,bool)
        items.append(dict(index=i, prompt=prompts[i], predicted_iou_score=scores[i],
            pixels=int(mask.sum()), rgb_box=box(mask), measured_depth_pixels=int(len(depths)),
            rgb_depth_coverage_fraction=float(np.count_nonzero(mask & valid)/max(1,np.count_nonzero(mask))),
            depth_median_m=None if not len(depths) else float(np.median(depths)),
            depth_p05_p95_m=None if not len(depths) else np.percentile(depths,[5,95]).tolist(),
            measured_centroid_color_camera_m=None if not len(xyz) else xyz.mean(axis=0).tolist(),
            measured_extent_color_camera_m=None if not len(xyz) else np.ptp(xyz,axis=0).tolist(),
            baseline_agreement_iou=overlap(mask & valid,reference & valid) if target is not None else None,
            full_image_baseline_agreement_iou=overlap(mask,reference) if target is not None else None,
            agreement_with_baseline_labels={str(label):overlap(mask & valid,(baseline['labels']==label) & valid)
                for label in np.unique(baseline['labels']) if label>=0}))
    return items


def export(folder, frame, baseline, masks, prompts, scores, metadata, calibration):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(folder/'rgb.png'),cv2.cvtColor(frame.color,cv2.COLOR_RGB2BGR))
    np.save(folder/'depth_meters.npy',frame.depth)
    if baseline['result'] is not None:
        np.save(folder/'depth_labels_native.npy',baseline['result'].object_labels)
    np.save(folder/'baseline_labels_rgb.npy',baseline['labels'])
    np.save(folder/'sam_masks.npy',masks)
    for i,mask in enumerate(masks):
        cv2.imwrite(str(folder/f'sam_mask_{i:02d}.png'),mask.astype(np.uint8)*255)
        bounds = box(mask)
        if bounds:
            x0,y0,x1,y1 = bounds
            cv2.imwrite(str(folder/f'crop_{i:02d}.png'),cv2.cvtColor(frame.color[y0:y1+1,x0:x1+1],cv2.COLOR_RGB2BGR))
    left,right = panels(frame,baseline,masks,sam_ran=metadata.get('engine')!='depth_only')
    cv2.imwrite(str(folder/'baseline.png'),left)
    cv2.imwrite(str(folder/'sam.png'),right)
    cv2.imwrite(str(folder/'comparison.png'),np.hstack((left,right)))
    report = dict(**metadata,timestamp_ns=int(frame.timestamp_ns),
        rgb_sha256=hashlib.sha256(frame.color.tobytes()).hexdigest(),
        depth_sha256=hashlib.sha256(frame.depth.tobytes()).hexdigest(),
        calibration=calibration_dict(calibration), baseline_error=baseline['error'],
        plane_model=None if baseline['result'] is None else baseline['result'].plane_model,
        baseline_segmentation_ms=baseline['segmentation_ms'], projection_ms=baseline['projection_ms'],
        objects=measurements(frame,baseline,masks,prompts,scores),
        metric_note='Agreement with the depth baseline is not ground-truth accuracy; scores are model predictions.')
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    return report
