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
import torch.nn as nn
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate

# 创建一个100x100x3的随机数组
x = np.random.rand(100, 100, 3)
print(x.shape)


temp = x[1:]
print(temp.shape)


temp = x[:, 1:]
print(temp.shape)


temp = x[:, :, 1:]
print(temp.shape)
