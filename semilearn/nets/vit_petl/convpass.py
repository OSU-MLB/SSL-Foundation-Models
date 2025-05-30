import torch
from torch import nn
from .utils import init_weight

class ConvPass(nn.Module):
    def __init__(self, dim, peft_config=None):
        super().__init__()

        self.bottleneck = peft_config.convpass_bottleneck
        # import pdb; pdb.set_trace()
        self.adapter_conv = nn.Conv2d(self.bottleneck, self.bottleneck, 3, 1, 1)
        if peft_config.convpass_xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(self.bottleneck, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

        self.adapter_down = nn.Linear(dim, self.bottleneck)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(self.bottleneck, dim)  # equivalent to 1 * 1 Conv
        init_weight(self.adapter_down, self.adapter_up, peft_config.convpass_init)


        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.scale = peft_config.convpass_scaler
        assert peft_config.crop_size == 224
        self.patch_num = peft_config.crop_size // peft_config.patch_size
        self.peft_config = peft_config

    def forward(self, x):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        x_patch = x_down[:, 1:].reshape(B, self.patch_num, self.patch_num, self.bottleneck).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, self.patch_num * self.patch_num, self.bottleneck)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.bottleneck).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.bottleneck)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv

        x_up = x_up * self.scale
        return x_up


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
