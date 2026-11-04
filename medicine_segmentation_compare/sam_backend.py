"""Prompted SAM 2.1, inspired by Hello Robot's sam2_utils.py (see ATTRIBUTION.md).

Independent image implementation: batched prompts, predicted-quality mask
selection, explicit timings. No model/robot actions run on import.
"""

from time import perf_counter
from pathlib import Path
import os

import numpy as np


class SamSegmenter:
    def __init__(self, model_id='facebook/sam2.1-hiera-tiny', device='auto', revision=None):
        started = perf_counter()
        print(f'Loading SAM dependencies and {model_id}...',flush=True)
        os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parent/'models'/'huggingface'))
        import torch
        from transformers import Sam2Model, Sam2Processor
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        self.torch = torch
        torch.set_num_threads(4)
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu')
        print(f'Loading {model_id} on {self.device}...',flush=True)
        if revision is None and model_id == 'facebook/sam2.1-hiera-tiny':
            revision = 'de431c4043854a71d8101e17995dfe596bf101a5'
        location = model_id
        if not Path(model_id).is_dir():
            try:
                cached = snapshot_download(model_id,revision=revision,local_files_only=True)
                if (Path(cached)/'model.safetensors').exists():
                    location = cached
            except LocalEntryNotFoundError:
                pass
        options = {} if Path(location).is_dir() else dict(revision=revision)
        self.processor = Sam2Processor.from_pretrained(location,**options)
        model,self.loading_info = Sam2Model.from_pretrained(location,output_loading_info=True,**options)
        if self.loading_info.get('missing_keys'):
            raise RuntimeError(f'Checkpoint is missing model weights: {self.loading_info["missing_keys"]}')
        self.model = model.to(self.device).eval()
        self.revision = Path(location).name if location != model_id else getattr(self.model.config,'_commit_hash',None)
        self.model_id = model_id
        self.load_ms = (perf_counter()-started)*1000
        print(f'Model ready in {self.load_ms/1000:.1f}s',flush=True)

    def segment(self, rgb, prompts):
        if not prompts:
            return np.zeros((0,*rgb.shape[:2]), bool), [], dict(preprocess_ms=0, inference_ms=0, postprocess_ms=0)
        kinds = {p['kind'] for p in prompts}
        if len(kinds) != 1:
            raise ValueError('Use either boxes or points in one batch')
        started = perf_counter()
        if kinds == {'box'}:
            kwargs = dict(input_boxes=[[p['coordinates'] for p in prompts]])
        else:
            kwargs = dict(input_points=[[[p['coordinates']] for p in prompts]],
                          input_labels=[[[1] for _ in prompts]])
        inputs = self.processor(images=rgb, return_tensors='pt', **kwargs).to(self.device)
        if self.device == 'cuda':
            self.torch.cuda.synchronize()
        prepared = perf_counter()
        with self.torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=True)
        if self.device == 'cuda':
            self.torch.cuda.synchronize()
        inferred = perf_counter()
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
        scores = outputs.iou_scores[0].detach().cpu().numpy()
        if not np.isfinite(scores).all():
            raise RuntimeError('SAM returned nonfinite quality scores')
        winners = scores.argmax(axis=-1)
        selected = np.stack([masks[i,int(w)].numpy().astype(bool) for i,w in enumerate(winners)])
        quality = [float(scores[i,w]) for i,w in enumerate(winners)]
        return selected, quality, dict(preprocess_ms=(prepared-started)*1000,
                                      inference_ms=(inferred-prepared)*1000,
                                      postprocess_ms=(perf_counter()-inferred)*1000)
