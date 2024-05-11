import os
import torch
from torch import nn
from prompts.vpt_vit_mae import (
    vit_base_patch16
)
from .mlp import MLP
from torchvision import models
import torch.nn.functional as F
from segmentation_head.deeplab_head import DeepLabHead

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
        
        # self.enc = models.resnet18(pretrained=True)
        # self.enc.fc = torch.nn.Identity()
        # self.feat_dim = 512

    def setup_head(self, cfg):
        self.head = models.segmentation.deeplabv3.DeepLabHead(self.feat_dim, 21)

    def forward(self, x):
        input_shape = x.shape[-2:]

        if self.frozen_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim
        
        z = self.head(x.unsqueeze(2).unsqueeze(3))
        z = F.interpolate(z, size=input_shape, mode='bilinear', align_corners=False)

        return x, z

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x
