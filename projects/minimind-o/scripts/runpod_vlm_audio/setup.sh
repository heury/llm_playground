#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120
export OMNI_RUN_ROOT=${OMNI_RUN_ROOT:-/workspace/vlm-audio-20261109}
python -m pip install --break-system-packages 'transformers==4.57.6' 'huggingface_hub>=0.34,<1' 'pyarrow>=19' 'Pillow==11.3.0' soundfile librosa onnxruntime funasr addict modelscope datasets
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name()); print((torch.randn(256,256,device="cuda")@torch.randn(256,256,device="cuda")).sum().item())'
python "$OMNI_RUN_ROOT/repo/scripts/runpod_vlm_audio/prepare.py"
python "$OMNI_RUN_ROOT/repo/scripts/runpod_vlm_audio/train.py"
python "$OMNI_RUN_ROOT/repo/scripts/runpod_vlm_audio/evaluate.py"
