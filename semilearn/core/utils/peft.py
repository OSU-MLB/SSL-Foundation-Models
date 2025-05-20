from dotwiz import DotWiz
import copy

_DEFAULT_PEFT_CONFIG = {
    "ft_attn_module": None,
    "ft_attn_mode": "parallel",
    "ft_attn_ln": "before",
    "ft_mlp_module": None,
    "ft_mlp_mode": "parallel",
    "ft_mlp_ln": "before",
    "adapter_bottleneck": 64,
    "adapter_init": "lora_kaiming",
    "adapter_scaler": 0.1,
    "convpass_bottleneck": 8,
    "convpass_xavier_init": False,
    "convpass_init": "lora_xavier",
    "convpass_scaler": 10,
    "vpt_mode": None,
    "vpt_num": 10,
    "vpt_layer": None,
    "vpt_dropout": 0.1,
    "vqt_num": 0,
    "ssf": False,
    "lora_bottleneck": 0,
    "fact_dim": 8,
    "fact_type": None,
    "fact_scaler": 1.0,
    "repadapter_bottleneck": 8,
    "repadapter_init": "lora_xavier",
    "repadapter_scaler": 1,
    "repadapter_group": 2,
    "bitfit": False,
    "attention_type": "full",
    "ln": False,
    "difffit": False,
    "freeze_backbone": False,
}

def get_peft_config(peft_config):
    if peft_config is None:
        return DotWiz(_DEFAULT_PEFT_CONFIG)
    else:
        _peft_config = copy.deepcopy(_DEFAULT_PEFT_CONFIG)
        _peft_config.update(peft_config)
        return DotWiz(_peft_config)
