import copy
import timm
import torch

from timm.layers import PatchEmbed
import torch.nn as nn

from .vit_petl.vit import *

TUNE_MODULES = ['ft_attn_module', 'ft_mlp_module', 'head', 'vpt', 'ssf_scale', 'ssf_shift', 'lora', 'fact', 'vqt', 'difffit']


class TimmViTWrapper(nn.Module):
    def __init__(self, model):
        super(TimmViTWrapper, self).__init__()
        self.model = model
        self.num_features = self.model.num_features
        # self._named_parameters = named_parameters
    
    # def named_parameters(self):
    #     return zip(self._named_parameters.keys(), self._named_parameters.values())

    # Currently only support ViT based backbone
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.forward_features(x)
        pooled_x = self.model.pool(x)
        return pooled_x
    
    def forward_head(self, pooled_x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.model.fc_norm(pooled_x)
        x = self.model.head_drop(x)
        return x if pre_logits else self.model.head(x)

    def forward(self, x, only_fc=False, only_feat=False):
        assert not (only_fc and only_feat), "only_fc and only_feat cannot be True at the same time"

        if (not only_fc) and (not only_feat):
            feat = self.forward_features(x)
            logits = self.forward_head(feat)
            return_dict = dict(feat=feat, logits=logits)
            return return_dict
        elif only_feat:
            feat = self.forward_features(x)
            return feat
        else:
            logits = self.forward_head(x)
            return logits


def timm_builder(name, peft_config, vit_config, pretrained=True, pretrained_path='', **kwargs):
    print(f"Buliding timm model with name: {name}, `pretrained`: {pretrained}, `pretrained_path`: {pretrained_path} and kwargs: {kwargs}")

    # Sanity check
    assert pretrained

    ## TODO: Refactor img_size
    vit_config = {**vit_config, **kwargs, 'img_size': 224}
    print(f"vit_config: {vit_config}, peft_config: {peft_config}")
    
    num_classes = kwargs['num_classes']
    if name == 'vit_base_patch16_224.augreg_in21k':
        print("Using in21k model")
        model = timm.create_model("vit_base_patch16_224_in21k_petl", pretrained=False, peft_config=peft_config, **vit_config)
        model.load_pretrained(
            'pretrain_weight/vit_base_patch16_224_augreg_in21k.bin')
        model.reset_classifier(num_classes)
    elif name == 'vit_base_patch14_reg4_dinov2.lvd142m':
        print("Using dino model")
        model = timm.create_model("vit_base_patch14_dinov2_petl", 
                                  pretrained=False,
                                  peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained(
            'pretrain_weight/vit_base_patch14_reg4_dinov2_lvd142m.bin')
        model.reset_classifier(num_classes)
    elif name == 'vit_base_patch16_clip_224.openai':
        print("Using clip model")
        model = timm.create_model("vit_base_patch16_clip_224_petl", 
                                  pretrained=False,
                                  peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained(
            'pretrain_weight/vit_base_patch16_clip_224_openai.bin')
        model.reset_classifier(num_classes)
    elif name == 'ViT-B-16-SigLIP':
        # wget https://huggingface.co/timm/ViT-B-16-SigLIP/resolve/main/open_clip_pytorch_model.bin?download=true
        # mv open_clip_pytorch_model.bin\?download\=true siglip.bin
        model = timm.create_model("vit_base_patch16_siglip_224_petl",
                                    pretrained=False,
                                    peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained('pretrain_weight/siglip.bin')
        model.reset_classifier(num_classes)
    elif name == 'vit_large_patch14_clip_224.openai':
        # wget https://huggingface.co/timm/vit_large_patch14_clip_224.openai/resolve/main/pytorch_model.bin?download=true
        # mv pytorch_model.bin\?download\=true vit_large_patch14_clip_224.openai.binf
        model = timm.create_model("vit_large_patch14_clip_224_petl",
                                    pretrained=False,
                                    peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained('pretrain_weight/vit_large_patch14_clip_224.openai.bin')
        model.reset_classifier(num_classes)
    elif name == 'vit_large_patch14_reg4_dinov2.lvd142m':
        # wget https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m/resolve/main/pytorch_model.bin?download=true
        # mv pytorch_model.bin\?download\=true vit_large_patch14_reg4_dinov2.lvd142m.bin
        model = timm.create_model("vit_base_patch14_dinov2_large_petl",
                                    pretrained=False,
                                    peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained('pretrain_weight/vit_large_patch14_reg4_dinov2.lvd142m.bin')
        model.reset_classifier(num_classes)
    elif name == 'vit_large_patch14_dinov2.lvd142m':
        # https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m/resolve/main/pytorch_model.bin?download=true
        # mv pytorch_model.bin\?download\=true vit_large_patch14_dinov2.lvd142m.bin
        model = timm.create_model("vit_base_patch14_dinov2_large_petl",
                                    pretrained=False,
                                    peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained('pretrain_weight/vit_large_patch14_dinov2.lvd142m.bin')
    elif name == 'open_clip_ViT-B-16':
        model = timm.create_model("vit_base_patch16_metaclip_224_petl",
                                    pretrained=False,
                                    peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained('pretrain_weight/open_clip_ViT-B-16.bin')
    elif name == 'metaclip_400m':
        # wget https://huggingface.co/timm/vit_base_patch16_clip_224.metaclip_400m/resolve/main/open_clip_pytorch_model.bin?download=true
        # mv open_clip_pytorch_model.bin\?download\=true vit_base_patch16_clip_224.metaclip_400m.bin
        model = timm.create_model("vit_base_patch16_metaclip_224_petl",
                                    pretrained=False,
                                    peft_config=peft_config,
                                    **vit_config)
        model.load_pretrained('pretrain_weight/vit_base_patch16_clip_224.metaclip_400m.bin')
    else:
        raise ValueError(f'Unknown model name: {name}')
    
    if "method_name" in peft_config:
        print("Performing PEFT timm model building")
        model, tune_parameters = petl_builder(model, peft_config)
    
    if pretrained_path != '' and pretrained_path != None:
        print(f"Loading pretrained model from {pretrained_path}")
        state_dict = torch.load(pretrained_path)['model']
        state_dict = {k.replace('module.model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    return TimmViTWrapper(model)


def petl_builder(model, peft_config):
    if peft_config.freeze_backbone:
        print("Freezing backbone")
        _TUNE_MODULES = ['head']
    else:
        print("Not freezing backbone")
        _TUNE_MODULES = copy.deepcopy(TUNE_MODULES)

        if peft_config.bitfit or peft_config.difffit:
            _TUNE_MODULES.append('bias')

        if peft_config.ln or peft_config.difffit:
            _TUNE_MODULES.append('norm')

    tune_parameters = {}

    for name, parameter in model.named_parameters():
        if any(m in name for m in _TUNE_MODULES):
            parameter.requires_grad = True
            # tune_parameters.append(parameter)
            tune_parameters[name] = parameter
        else:
            parameter.requires_grad = False

    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    model_grad_params_no_head = sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad and 'head' not in n)
    print("Total Parameters: {0}\t Gradient Parameters: {1}\t Gradient Parameters No Head: {2}".format(
        model_total_params, model_grad_params, model_grad_params_no_head))
    print(f"total tuned percent:{(model_grad_params / model_total_params * 100):.2f} %")
    print(f"total tuned percent no head:{(model_grad_params_no_head / model_total_params * 100):.2f} %")

    # todo ping the tune_parameters are the parameters added to optimizer
    return model, tune_parameters
