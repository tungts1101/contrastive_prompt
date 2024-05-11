import torch
from torch import nn
from models.contrastive_segmentation import (
    ContrastiveSegmentation
)
from dataloaders import pascal_voc
from torchmetrics.classification import MulticlassJaccardIndex
import utils

class Config:
    def __init__(self):
        self.data_dir = '/media/ellen/datasets'
        self.batch_size = 64
        self.workers = 4

if __name__ == '__main__':
    device = torch.device('cuda')
    model = ContrastiveSegmentation({}).to(device)
    utils.log_model_info(model)
    cfg = Config()
    trainloader = pascal_voc.get_train_dataloader(cfg)
    valloader = pascal_voc.get_val_dataloader(cfg)
    metrics = MulticlassJaccardIndex(num_classes=21).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 30
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            embed, z = model(x)
            loss = loss_fn(z, y.squeeze())
            loss.backward()
            optimizer.step()
            
            print(f"epoch: {epoch}, iter: {batch_idx}, loss: {loss.item()}")
        
        with torch.no_grad():
            for batch in valloader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                embed, z = model(x)
                iou = metrics.update(z, y.squeeze())
            print(f"mIoU: {metrics.compute()}")
            metrics.reset()
      