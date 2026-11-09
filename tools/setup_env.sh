#!/usr/bin/env bash
# Create .venv with the pinned inference + evaluation stack (CUDA 12.x driver).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install accelerate "diffusers>=0.31" easydict einops ftfy imageio imageio-ffmpeg "numpy>=1.23.5,<2" \
  "opencv-python-headless>=4.9" scipy "tokenizers>=0.20.3" "transformers>=4.56,<4.57" tqdm pyyaml pandas \
  pytest safetensors sentencepiece "huggingface_hub[cli]" hf_transfer
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e .
python -c "import torch, flash_attn; print('torch', torch.__version__, 'flash-attn', flash_attn.__version__)"
