import cv2
import numpy as np
import argparse
import os
import datetime
import torch
import re
from scipy import io
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from scipy.ndimage import zoom
import torch
import torch.nn as nn
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate

from src.model.common import deconv3d_2x
from src.model.common import Interpolate_trilinear

# 定义输入特征图大小
in_channels = 3
in_depth, in_height, in_width = 10, 10, 10
# 创建随机输入特征图
input = torch.randn(2, in_channels, in_depth, in_height, in_width)
# output = nn.functional.interpolate(input, scale_factor=2, mode='trilinear')
# 上采样模块
func_upsample = Interpolate_trilinear(2)
# 前向传播
output = func_upsample(input)
# 输出结果大小
batch_size, out_channels, out_depth, out_height, out_width = output.size()
print("Output Size:", batch_size, out_channels, out_depth, out_height, out_width)




