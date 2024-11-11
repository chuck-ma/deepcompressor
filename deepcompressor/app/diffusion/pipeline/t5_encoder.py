from huggingface_hub import hf_hub_download
import os
import copy
import torch
import torch.nn as nn
import transformers.utils, transformers.modeling_utils
from hqq.core.quantize import HQQLinear
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel as OriginalT5EncoderModel


from typing import Optional, Union, Dict, List, Any


class T5EncoderModel(OriginalT5EncoderModel):
    _torch_dtype = torch.float32
    _hqq_4bit_compute_dtype = torch.float32

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for arg in args:
            if isinstance(arg, torch.dtype):
                self._torch_dtype = arg
                break
        for k, v in kwargs.items():
            if k == "device" and isinstance(v, torch.dtype):
                self._torch_dtype = v
                break
        return self

    @property
    def dtype(self) -> torch.dtype:
        return self._torch_dtype

    @property
    def hqq_4bit_compute_dtype(self) -> torch.dtype:
        return self._hqq_4bit_compute_dtype

    @hqq_4bit_compute_dtype.setter
    def hqq_4bit_compute_dtype(self, value: torch.dtype):
        for module in self.modules():
            if isinstance(module, HQQLinear):
                module.compute_dtype = value
                module.meta["compute_dtype"] = value
        self._hqq_4bit_compute_dtype = value

    def state_dict(self, *args, **kwargs):
        global _in_state_dict_fn
        in_state_dict_fn = _in_state_dict_fn
        _in_state_dict_fn = True
        result = super().state_dict(*args, **kwargs)
        _in_state_dict_fn = in_state_dict_fn
        return result

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        hqq_4bit_compute_dtype: Optional[torch.dtype] = None,
        **kwargs_ori,
    ) -> "T5EncoderModel":
        kwargs = copy.deepcopy(kwargs_ori)
        cache_dir = kwargs.pop("cache_dir", None)
        subfolder = kwargs.pop("subfolder", None)
        torch_dtype = kwargs.pop("torch_dtype", T5EncoderModel._torch_dtype)
        if hqq_4bit_compute_dtype is None:
            hqq_4bit_compute_dtype = torch_dtype
        model_config: Dict[str, Any] = T5Config.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            cache_dir=cache_dir,
            **kwargs,
        )
        quant_config: Dict[str, Any] = getattr(
            model_config, "quantization_config", {"quant_method": None}
        )
        is_hqq_quant = quant_config.pop("quant_method") == "hqq"
        if not is_hqq_quant:
            return super(T5EncoderModel, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs_ori
            )
        if hasattr(model_config, "quantization_config"):
            delattr(model_config, "quantization_config")
        with torch.device("meta"):
            model = T5EncoderModel._from_config(
                config=model_config,
                torch_dtype=torch_dtype,
            )
        model._torch_dtype = torch_dtype
        model._hqq_4bit_compute_dtype = hqq_4bit_compute_dtype

        modules = {name: module for name, module in model.named_modules()}
        parameters = {name: param for name, param in model.named_parameters()}
        for name, param in parameters.items():
            parent_name, param_name = (
                ".".join(name.split(".")[:-1]),
                name.split(".")[-1],
            )
            module = modules[parent_name]
            if not isinstance(module, nn.Linear):
                dtype = param.dtype
                if "float" in repr(dtype):
                    dtype = torch_dtype
                param = torch.empty_like(param, dtype=dtype, device="cuda")
                setattr(module, param_name, nn.Parameter(param))
        # model.quantization_method = "hqq" # not to set it, then we can use `.to(device)` in cpu_offload
        for name, module in modules.items():
            if isinstance(module, nn.Linear):
                parent_name, linear_name = (
                    ".".join(name.split(".")[:-1]),
                    name.split(".")[-1],
                )
                hqq_layer = HQQLinear(
                    None,  # torch.nn.Linear or None
                    quant_config=quant_config,  # quantization configuration
                    compute_dtype=hqq_4bit_compute_dtype,  # compute dtype
                    device="cuda",  # cuda device
                    initialize=True,  # Use False to quantize later
                    del_orig=True,  # if True, delete the original layer
                )
                hqq_layer.weight = torch.empty(
                    1, 1, dtype=torch_dtype, device="cuda"
                )  # Placeholder
                del module.weight
                del module.bias

                if parent_name == "":
                    parent_module = model
                else:
                    parent_module = modules[parent_name]
                setattr(parent_module, linear_name, hqq_layer)
        weight_path = pretrained_model_name_or_path
        if subfolder is not None:
            weight_path += "/" + subfolder
        weight_name = kwargs.pop("weight_name", "model.safetensors")
        weight_path += "/" + weight_name
        if not os.path.exists(weight_path):
            weight_path = hf_hub_download(
                pretrained_model_name_or_path,
                weight_name,
                subfolder=subfolder,
                cache_dir=cache_dir,
            )
        model_state_dict = load_file(weight_path, device="cuda")

        def make_cast_forward(self):
            forward_ori = self.forward

            def forward(x):
                return forward_ori(x.to(model._hqq_4bit_compute_dtype)).type_as(x)

            self.forward = forward
            return self

        model.load_state_dict(model_state_dict, strict=False)
        for name, module in model.named_modules():
            if isinstance(module, HQQLinear):
                state_dict = {
                    k.split(name + ".")[-1]: v
                    for k, v in model_state_dict.items()
                    if k.startswith(name + ".")
                }
                module.load_state_dict(state_dict)
                make_cast_forward(module)
                module.compute_dtype = hqq_4bit_compute_dtype
                module.meta["compute_dtype"] = hqq_4bit_compute_dtype

                del state_dict
        del model_state_dict
        torch.cuda.empty_cache()

        return model


text_encoder_path = "HighCWu/FLUX.1-dev-4bit"
dtype = torch.bfloat16

text_encoder_2: T5EncoderModel = T5EncoderModel.from_pretrained(
    text_encoder_path,
    subfolder="text_encoder_2",
    torch_dtype=dtype,
)
