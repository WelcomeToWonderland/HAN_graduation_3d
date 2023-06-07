import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv_3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        """
        scale ： 2的整数次幂
        
        不使用nn.PixelShuffle(scale)，一步完成上采样，而是使用多个nn.PixelShuffle(2)达成效果（这种方法，又称为深度上采样）
        1、减少计算量（不理解）
        2、特征学习：学习分辨率提升的过程中的特征（与PixelShuffle搭配的conv卷积层）
        """
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                """
                nn.PixelShuffle(r)
                应用于二维图像
                将通道数缩小r^2倍，将H和W维度扩大r倍
                """
                m.append(nn.PixelShuffle(2))
                """
                下面没有生效
                """
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Upsampler_3d(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        """
        scale ： 2的整数次幂
        PixelShuffle3d 没有被是实现的函数
        改用nn.Upsample，一步完成上采样
        """
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 8 * n_feats, 3, bias))
                m.append(nn.PixelShuffle3d(2))
                # m.append(conv(n_feats, n_feats, 3, bias))
                # m.append(nn.Upsample(scale_factor=2))
        elif scale == 3:
            # m.append(conv(n_feats, 27 * n_feats, 3, bias))
            # m.append(nn.PixelShuffle3d(3))
            m.append(conv(n_feats, n_feats, 3, bias))
            m.append(nn.Upsample(scale_factor=3))
        else:
            raise NotImplementedError

        super(Upsampler_3d, self).__init__(*m)

