"""Seeded outline refiner architecture, with original checkpoint parameter names."""

import torch
from torch import nn
from torch.nn import functional


class Block(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(True),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.BatchNorm2d(outputs), nn.ReLU(True))

    def forward(self, inputs):
        return self.net(inputs)


class SeededUNet(nn.Module):
    def __init__(self, width=24, channels=4):
        super().__init__()
        self.down1 = Block(channels, width)
        self.down2 = Block(width, width*2)
        self.down3 = Block(width*2, width*4)
        self.middle = Block(width*4, width*8)
        self.up3 = nn.ConvTranspose2d(width*8, width*4, 2, 2)
        self.dec3 = Block(width*8, width*4)
        self.up2 = nn.ConvTranspose2d(width*4, width*2, 2, 2)
        self.dec2 = Block(width*4, width*2)
        self.up1 = nn.ConvTranspose2d(width*2, width, 2, 2)
        self.dec1 = Block(width*2, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, inputs):
        down1 = self.down1(inputs)
        down2 = self.down2(functional.max_pool2d(down1, 2))
        down3 = self.down3(functional.max_pool2d(down2, 2))
        middle = self.middle(functional.max_pool2d(down3, 2))
        up3 = self.dec3(torch.cat([self.up3(middle), down3], 1))
        up2 = self.dec2(torch.cat([self.up2(up3), down2], 1))
        up1 = self.dec1(torch.cat([self.up1(up2), down1], 1))
        return self.head(up1)
