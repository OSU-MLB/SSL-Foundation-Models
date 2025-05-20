import torch
import torch.nn as nn
from functools import reduce
from operator import mul
import math
from torch.nn.modules.utils import _pair


class VPT(nn.Module):

    def __init__(self, peft_config, depth, patch_size, embed_dim):
        super().__init__()
        self.peft_config = peft_config
        self.depth = depth
        if peft_config.vpt_mode == 'shallow':
            prompt_layer = 1
        elif peft_config.vpt_mode == 'deep':
            if peft_config.vpt_layer:
                prompt_layer = peft_config.vpt_layer
            else:
                prompt_layer = depth

        else:
            raise ValueError
        val = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size), 1) + embed_dim))
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            prompt_layer, peft_config.vpt_num, embed_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        self.prompt_dropout = nn.Dropout(peft_config.vpt_dropout)


    def retrieve_prompt(self, index, batch_size):
        if self.peft_config.vpt_layer:
            index = index - (self.depth - self.peft_config.vpt_layer)
            if index < 0:
                return None
        if index < len(self.prompt_embeddings):
            return self.prompt_dropout(self.prompt_embeddings[index]).expand(batch_size, -1, -1)
        else:
             return None
