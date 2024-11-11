# 如果代码有改动，需要重新安装
pip install -e .

python -m deepcompressor.app.diffusion.ptq configs/model/flux.1-custom.yaml configs/svdquant/int4.yaml
