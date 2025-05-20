from torch import nn
from .utils import init_weight


class RepAdapter(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            dim,
            peft_config,
    ):
        super().__init__()
        self.conv_A = nn.Conv1d(dim, peft_config.repadapter_bottleneck, 1, groups=1, bias=True)
        self.conv_B = nn.Conv1d(peft_config.repadapter_bottleneck, dim, 1, groups=peft_config.repadapter_group, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.groups = peft_config.repadapter_group
        self.scale = peft_config.repadapter_scaler

        init_weight(self.conv_A, self.conv_B, peft_config.repadapter_init)
        self.peft_config = peft_config

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale + x
        x = x.transpose(1, 2).contiguous()
        return x
