''' network architecture for deep_down '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import models.archs.arch_util as arch_util
import numpy as np
import math
import pdb
from torch.nn.modules.utils import _pair
from torch.autograd import Variable

class CNN_downsampling(nn.Module):
    def __init__(self, input_channels=4, kernel_size=3):
        super(CNN_downsampling, self).__init__()
        padding = 1
        layers = []
        layers.append(nn.Conv2d(input_channels, 4*input_channels, kernel_size, 2, 1, bias=False))
        layers.append(nn.Conv2d(in_channels=4*input_channels, \
            out_channels=4*input_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(nn.Conv2d(4*input_channels, 4*input_channels, kernel_size, 2, 1, bias=False))
        layers.append(nn.Conv2d(in_channels=4*input_channels, \
            out_channels=4*input_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        layers.append(nn.Conv2d(in_channels=4*input_channels, \
            out_channels=4*input_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.InstanceNorm2d(4*input_channels))
        
        self.simple_block = nn.Sequential(*layers)
        self._initialize_weights()


    def forward(self, x):
        out = self.simple_block(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
