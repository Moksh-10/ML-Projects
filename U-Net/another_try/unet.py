import copy
import os
import random
import shutil
import zipfile
from math import atan2, cos, sin, sqrt, pi, log

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from numpy import linalg as LA
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        a = self.conv(x)
        b = self.pool(a)
        return a, b


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class unet(nn.Module):
    def __init__(self, in_ch, num_cl):
        super().__init__()
        self.dc1 = down(in_ch, 64)
        self.dc2 = down(64, 128)
        self.dc3 = down(128, 256)
        self.dc4 = down(256, 512)

        self.bn = double_conv(512, 1024)

        self.uc1 = up(1024, 512)
        self.uc2 = up(512, 256)
        self.uc3 = up(256, 128)
        self.uc4 = up(128, 64)

        self.out = nn.Conv2d(64, num_cl, kernel_size=1)

    def forward(self, x):
        d1, p1 = self.dc1(x)
        d2, p2 = self.dc2(p1)
        d3, p3 = self.dc3(p2)
        d4, p4 = self.dc4(p3)

        b = self.bn(p4)

        u1 = self.uc1(b, d4)
        u2 = self.uc2(u1, d3)
        u3 = self.uc3(u2, d2)
        u4 = self.uc4(u3, d1)

        out = self.out(u4)
        return out






