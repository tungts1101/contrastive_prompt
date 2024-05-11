from torchvision import transforms
from .operators import FixedResize, ColorTransform, ToTensor, Normalization

def get_train_color_transform():
    return transforms.Compose(
        [FixedResize(), ColorTransform(), ToTensor(), Normalization()]
    )

def get_val_transform():
    return transforms.Compose([FixedResize(), ToTensor(), Normalization()])
