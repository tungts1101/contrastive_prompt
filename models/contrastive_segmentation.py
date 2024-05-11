import os
import torch
from torch import nn
from prompts.vpt_vit_mae import (
    vit_base_patch16
)
from segmentation_head.deeplab_head import (
    DeepLabHead
)
from .mlp import MLP

class ContrastiveSegmentation(nn.Module):
    def __init__(self, cfg, load_pretrain=True):
        super(ContrastiveSegmentation, self).__init__()
        self.cfg = cfg

        self.frozen_enc = False
        
        self.build_backbone(None)
        self.setup_head(cfg)
    
    def build_backbone(self, prompt_cfg):
        ckpt = os.path.join('checkpoints', "mae_pretrain_vit_base.pth")
        checkpoint = torch.load(ckpt, map_location="cpu")
        state_dict = checkpoint['model']

        self.enc = vit_base_patch16({})
        self.enc.load_state_dict(state_dict, strict=False)
        self.enc.head = torch.nn.Identity()
        self.feat_dim = self.enc.embed_dim
        
        for k, p in self.enc.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False

    def setup_head(self, cfg):
        self.neck = MLP(64 * self.feat_dim, [256 * self.feat_dim])
        self.head = DeepLabHead(
            in_channels=self.feat_dim, 
            num_classes=21) # TODO: config num_classes

    def forward(self, x, return_feature=False):

        if self.frozen_enc and self.enc.training:
            self.enc.eval()
        x_enc = self.enc(x)  # batch_size x self.feat_dim

        if return_feature:
            return x_enc, x

        x = x_enc.flatten()
        x = self.neck(x)
        x = x.view(256, self.feat_dim)
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.head(x)

        return x_enc, x

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x
