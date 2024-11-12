from diffusers import FluxTransformer2DModel
import torch


def load_flux_model(
    model_path: str,
    load_from_file: bool = False,
    use_4bit: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> FluxTransformer2DModel:
    """
    加载FLUX模型，支持从单文件或预训练目录加载

    参数:
        model_path: 模型路径，可以是safetensors文件路径或预训练模型目录
        load_from_file: 是否从单个文件加载
        use_4bit: 是否使用4bit量化
        dtype: 模型计算精度
    """
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
    else:
        quantization_config = None

    if load_from_file:
        model = FluxTransformer2DModel.from_single_file(
            model_path, quantization_config=quantization_config, torch_dtype=dtype
        )
    else:
        model = FluxTransformer2DModel.from_pretrained(
            model_path, quantization_config=quantization_config, torch_dtype=dtype
        )

    return model
