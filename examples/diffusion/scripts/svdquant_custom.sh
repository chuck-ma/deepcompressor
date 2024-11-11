# 如果代码有改动，需要重新安装
pip install -e .

cd examples/diffusion
HF_ENDPOINT="https://hf-mirror.com" HF_HOME="/root/autodl-tmp/huggingface" MODELSCOPE_CACHE="/root/autodl-tmp/modelscope/hub" python -m deepcompressor.app.diffusion.ptq configs/model/flux.1-custom.yaml configs/svdquant/int4.yaml
