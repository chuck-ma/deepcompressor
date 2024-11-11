# 如果代码有改动，需要重新安装
pip install -e .

cd examples/diffusion

### Step 0: Environment Preparation
#### 下载 MJHQ-30K 数据集
HF_ENDPOINT="https://hf-mirror.com" HF_HOME="/root/autodl-tmp/huggingface" MODELSCOPE_CACHE="/root/autodl-tmp/modelscope/hub" python -c '
from datasets import load_dataset
dataset = load_dataset("playgroundai/MJHQ-30K")
'
### Step 1: Evaluation Baselines Preparation

HF_ENDPOINT="https://hf-mirror.com" HF_HOME="/root/autodl-tmp/huggingface" MODELSCOPE_CACHE="/root/autodl-tmp/modelscope/hub" python -m deepcompressor.app.diffusion.ptq configs/model/flux.1-custom.yaml --output-dirname reference


### Step 2: Calibration Dataset Preparation

HF_ENDPOINT="https://hf-mirror.com" HF_HOME="/root/autodl-tmp/huggingface" MODELSCOPE_CACHE="/root/autodl-tmp/modelscope/hub" python -m deepcompressor.app.diffusion.dataset.collect.calib \
    configs/model/flux.1-custom.yaml configs/collect/qdiff.yaml


### Step 3: Smooth Quantization

HF_ENDPOINT="https://hf-mirror.com" HF_HOME="/root/autodl-tmp/huggingface" MODELSCOPE_CACHE="/root/autodl-tmp/modelscope/hub" python -m deepcompressor.app.diffusion.ptq configs/model/flux.1-custom.yaml configs/svdquant/int4.yaml --save-model /root/autodl-tmp/flux.1-custom-svdquant-int4
