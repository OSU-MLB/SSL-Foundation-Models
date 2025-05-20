import math

import torch
import torch.nn as nn
from .utils import init_weight


class Adapter(nn.Module):
    def __init__(self, dim, peft_config, adapter_layernorm_option=None):

        super().__init__()
        self.n_embd = dim
        self.down_size = peft_config.adapter_bottleneck
        self.dropout = 0.1

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if peft_config.adapter_scaler == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(peft_config.adapter_scaler)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        init_weight(self.down_proj, self.up_proj, peft_config.adapter_init)
        self.peft_config = peft_config
    
    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
