import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from comparison import Comparison, SnapshotSource, box, export, measurements, overlap, proposals
from rgbd_types import Camera, Calibration, Frame


def fixture():
    depth_camera = Camera(2,1,np.array([[100.,0,0],[0,100.,0],[0,0,1]]),np.zeros(5),'none')
    color_camera = Camera(2,1,np.eye(3),np.zeros(5),'none')
    calibration = Calibration(color_camera,depth_camera,np.eye(4))
    frame = Frame(123456,np.array([[[1,2,3],[4,5,6]]],np.uint8),np.array([[1.,2.]],np.float32))
    return frame,calibration


class ComparisonTests(unittest.TestCase):
    def test_noncontiguous_labels_are_kept_in_prompts(self):
        labels = np.full((12,12),-1,np.int32)
        labels[1:5,2:6] = 7
        labels[7:11,8:12] = 12
        self.assertEqual([p['baseline_label'] for p in proposals(labels)],[7,12])
        self.assertEqual(proposals(labels)[0]['coordinates'],[2,1,5,4])

    def test_no_plane_still_produces_inspectable_projection(self):
        frame,c = fixture()
        with patch('comparison.segment_tabletop_objects',side_effect=RuntimeError('no plane')):
            baseline = Comparison(c).baseline(frame)
        self.assertEqual(baseline['error'],'no plane')
        self.assertTrue(np.all(baseline['labels']==-1))
        self.assertEqual(proposals(baseline['labels']),[])

    def test_mask_depth_uses_nearest_visible_point_not_hidden_background(self):
        frame,c = fixture()
        with patch('comparison.segment_tabletop_objects',side_effect=RuntimeError('no plane')):
            baseline = Comparison(c).baseline(frame)
        mask = np.array([[[True,False]]])
        values = measurements(frame,baseline,mask,[dict(kind='point',coordinates=[0,0])],[.9])[0]
        self.assertEqual(values['measured_depth_pixels'],1)
        self.assertEqual(values['depth_median_m'],1.)
        self.assertEqual(values['measured_centroid_color_camera_m'],[0.,0.,1.])

    def test_export_roundtrip_retains_metric_depth_rgb_and_calibration(self):
        frame,c = fixture()
        with patch('comparison.segment_tabletop_objects',side_effect=RuntimeError('no plane')):
            baseline = Comparison(c).baseline(frame)
        masks = np.array([[[True,False]],[[True,True]]])
        prompts = [dict(kind='point',coordinates=[0,0])]*2
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'frame'
            report = export(folder,frame,baseline,masks,prompts,[.8,.9],dict(engine='test'),c)
            np.testing.assert_array_equal(np.load(folder/'sam_masks.npy'),masks)
            self.assertEqual(len(report['objects']),2)
            with_source = SnapshotSource(folder)
            restored = with_source[0]
            np.testing.assert_array_equal(restored.color,frame.color)
            np.testing.assert_array_equal(restored.depth,frame.depth)
            np.testing.assert_array_equal(with_source.calibration.depth.K,c.depth.K)
            self.assertEqual(restored.timestamp_ns,frame.timestamp_ns)
            with_source.close()

    def test_agreement_is_intersection_over_union_and_empty_is_none(self):
        self.assertEqual(overlap(np.array([1,1,0],bool),np.array([0,1,1],bool)),1/3)
        self.assertIsNone(overlap(np.zeros(3,bool),np.zeros(3,bool)))
        self.assertIsNone(box(np.zeros((4,4),bool)))

    def test_sam_selects_quality_winner_per_object_and_batches_point_prompts(self):
        import torch
        from sam_backend import SamSegmenter
        class Inputs(dict):
            def to(self,device):
                return self
        predictions = torch.zeros((1,2,3,2,2),dtype=torch.bool)
        predictions[0,0,1,0,1] = True
        predictions[0,1,2,1,0] = True
        output = SimpleNamespace(pred_masks=predictions,
            iou_scores=torch.tensor([[[.1,.9,.2],[.2,.3,.8]]]))
        seen = {}
        class Processor:
            def __call__(self,**kwargs):
                seen.update(kwargs)
                return Inputs(original_sizes=torch.tensor([[2,2]]))
            def post_process_masks(self,masks,sizes):
                return [masks[0]]
        segmenter = SamSegmenter.__new__(SamSegmenter)
        segmenter.torch,segmenter.device = torch,'cpu'
        segmenter.processor = Processor()
        segmenter.model = lambda **kwargs: output
        prompts = [dict(kind='point',coordinates=[0,0]),dict(kind='point',coordinates=[1,1])]
        masks,scores,timings = segmenter.segment(np.zeros((2,2,3),np.uint8),prompts)
        self.assertEqual(masks.shape,(2,2,2))
        self.assertTrue(masks[0,0,1])
        self.assertTrue(masks[1,1,0])
        self.assertEqual(seen['input_points'],[[[[0,0]],[[1,1]]]])
        self.assertEqual(seen['input_labels'],[[[1],[1]]])
        np.testing.assert_allclose(scores,[.9,.8])


if __name__ == '__main__':
    unittest.main()
