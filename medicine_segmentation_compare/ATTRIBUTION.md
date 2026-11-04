# Attribution and scope

This experiment is inspired by Hello Robot's
[stretch4_grasping_demo](https://github.com/hello-robot/stretch4_grasping_demo),
particularly `sam2_utils.py` and the separation of RGB segmentation from depth
grasp geometry. Upstream is Copyright 2026 Hello Robot Inc., Apache License 2.0:
[license](https://github.com/hello-robot/stretch4_grasping_demo/blob/main/LICENSE),
[copyright notice](https://github.com/hello-robot/stretch4_grasping_demo/blob/main/LICENSE.md).

The implementation here is independently written against the documented Hugging
Face Sam2Model / Sam2Processor APIs. It does not vendor upstream code or load its
robot-control, fingertip calibration, Molmo, or motion modules.

Adaptations: multiple depth-box/click prompts; selecting the highest predicted
mask quality rather than always taking the first candidate; source/calibration
adapters for our bags/live debug bundles; quantitative paired exports; shared
playback; image-mode box re-prompting as an optional tracking baseline. This is
not upstream's SAM video-memory implementation.

SAM 2.1 models are distributed by Meta at
https://huggingface.co/facebook/sam2.1-hiera-tiny and
https://huggingface.co/facebook/sam2.1-hiera-large with their own license terms.

API reference used: https://huggingface.co/docs/transformers/model_doc/sam2
