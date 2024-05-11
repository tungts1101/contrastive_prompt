# Code referenced from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, norm_func=nn.BatchNorm2d):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36], norm_func),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_func(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_func=nn.BatchNorm2d):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            norm_func(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_func=nn.BatchNorm2d):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_func(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256, norm_func=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                norm_func(out_channels),
                nn.ReLU(),
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate, norm_func))

        modules.append(ASPPPooling(in_channels, out_channels, norm_func))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_func(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            print(x.shape)
            res.append(conv(x))
        print(f"here: {x.shape}")
        res = torch.cat(res, dim=1)
        return self.project(res)
