import torch
from models.contrastive_segmentation import (
    ContrastiveSegmentation
)
import utils

if __name__ == '__main__':
    device = torch.device('cuda')
    model = ContrastiveSegmentation({}).to(device)
    utils.log_model_info(model)
    imgs = torch.randn(64, 3, 224, 224).to(device)
    e, z = model(imgs)
    print(f"Encoder output shape: {e.shape}, Decoder output shape: {z.shape}")