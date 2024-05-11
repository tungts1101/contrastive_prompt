#!/usr/bin/env python3
"""
vit-moco-v3 with prompt
"""
import math
import torch
import torch.nn as nn
import torchvision as tv

from functools import partial, reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from foundations.vit_mae import VisionTransformer


class PromptedVisionTransformer(VisionTransformer):
    def __init__(self, prompt_cfg, **kwargs):
        super().__init__(**kwargs)
        self.prompt_cfg = prompt_cfg
        num_tokens = 50

        self.num_tokens = num_tokens

        # initiate prompt:
        val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, self.embed_dim))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
                x[:, :1, :],
                self.prompt_embeddings.expand(B, -1, -1),
                x[:, 1:, :]
            ), dim=1)
        return x

    def embeddings(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.blocks.eval()
            self.patch_embed.eval()
            self.pos_drop.eval()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.incorporate_prompt(x)
        for blk in self.blocks:
            x = blk(x)
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token # original
        outcome = self.fc_norm(x)

        return outcome


def build_model(model_type, prompt_cfg):
    if "vitb" in model_type:
        return vit_base_patch16(prompt_cfg)
    elif "vitl" in model_type:
        return vit_large_patch16(prompt_cfg)
    elif "vith" in model_type:
        return vit_huge_patch14(prompt_cfg)


def vit_base_patch16(prompt_cfg, **kwargs):
    model = PromptedVisionTransformer(
        prompt_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(prompt_cfg, **kwargs):
    model = PromptedVisionTransformer(
        prompt_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(prompt_cfg, **kwargs):
    model = PromptedVisionTransformer(
        prompt_cfg,
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


