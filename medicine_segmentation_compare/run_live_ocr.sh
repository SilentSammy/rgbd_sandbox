#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export USE_TF=0
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
exec env -u PYTHONPATH .venv/bin/python -u live_ocr_demo.py --live \
    --max-dimension 600 --scene-depth 1.1 --plane-min-inliers 300 \
    --model models/sam2.1-hiera-tiny "$@"
