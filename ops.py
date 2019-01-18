import torch.nn as nn
import math

class Conv_ReLU_Block(nn.Module):
    def __init__(self, nFeat, bias):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class ResBlock(nn.Module):
    def __init__(self, nFeat, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, kernel_size = 3):

        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(in_channels=nFeat, out_channels=nFeat, kernel_size=kernel_size, stride=1, padding=(kernel_size//2), bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(nFeat))
            if i == 0: modules_body.append(act)

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        if self.res_scale != 1:
            res = self.body(x).mul(self.res_scale)
        else:
            res = self.body(x)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, nFeat, bn=False, act=False, bias=True):
        super(Upsampler, self).__init__()

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                #modules.append(conv(nFeat, 4 * nFeat, 3, bias))
                modules.append(nn.Conv2d(in_channels=nFeat, out_channels=4 * nFeat, kernel_size=3, stride=1, padding=1, bias=bias))
                modules.append(nn.PixelShuffle(2))
                if bn: modules.append(nn.BatchNorm2d(nFeat))
                if act: modules.append(act())
        elif scale == 3:
            #modules.append(conv(nFeat, 9 * nFeat, 3, bias))
            modules.append(nn.Conv2d(in_channels=nFeat, out_channels=9 * nFeat, kernel_size=3, stride=1, padding=1, bias=bias))
            modules.append(nn.PixelShuffle(3))
            if bn: modules.append(nn.BatchNorm2d(nFeat))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        self.upsampler = nn.Sequential(*modules)

    def forward(self, x):
        return self.upsampler(x)

"""
random_uniform, random_gradual_**: These functions are used for training a scalable DNN by providing the probability distribution of each layer
"""

def random_uniform(elem_list):
    return elem_list

def random_gradual_01(elem_list, bias, weight):
    random_list = []

    for i in range(len(elem_list)):
        random_list.extend([elem_list[i]] * (bias + weight * i))

    return random_list

def random_gradual_02(elem_list, bias, weight):
    random_list = []

    for i in range(len(elem_list)):
        if i == len(elem_list) - 1:
            random_list.extend([elem_list[i]] * len(random_list))
        else:
            random_list.extend([elem_list[i]] * (bias + weight * i))

    return random_list

def random_gradual_03(elem_list):
    random_list = []

    if len(elem_list) == 1:
        random_list.extend([elem_list[0]])
    else:
        for i in range(len(elem_list)):
            if i == len(elem_list) - 1:
                random_list.extend([elem_list[i]] * len(random_list))
            else:
                random_list.extend([elem_list[i]] *  1)

    return random_list

def random_gradual_04(elem_list):
    random_list = []

    if len(elem_list) == 1:
        random_list.extend([elem_list[0]])
    else:
        for i in range(len(elem_list)):
            if i == len(elem_list) - 1:
                random_list.extend([elem_list[i]] * (len(random_list) - 2))
            else:
                random_list.extend([elem_list[i]] *  1)

    return random_list
