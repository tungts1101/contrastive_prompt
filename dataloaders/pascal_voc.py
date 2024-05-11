from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from transformations.transform_factory import (
    get_train_color_transform,
    get_val_transform,
)


class PascalVOCDataset(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        self.root_dataset = datasets.VOCSegmentation(
            root=cfg.data_dir, year="2012", image_set=split, download=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.root_dataset)

    def __getitem__(self, index):
        sample = self.root_dataset[index]
        image, mask = sample
        if self.transform:
            image, mask = self.transform(sample)
        return image, mask


def get_train_dataloader(cfg):
    dataset = PascalVOCDataset(cfg, transform=get_train_color_transform())
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )


def get_val_dataloader(cfg):
    dataset = PascalVOCDataset(cfg, split="val", transform=get_val_transform())
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
    )


def get_gt_val_dataset(cfg):
    return PascalVOCDataset(cfg, split="val", transform=None)
