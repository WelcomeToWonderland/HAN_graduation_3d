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
from src.PixelShuffle3D import PixelShuffle3D
from src.model import common

# dir = r'D:\workspace\dataset\USCT\original\HR'
# for filename in os.listdir(dir):
#     path = os.path.join(dir, filename)
#     file = io.loadmat(path)
#     img = file['f1']
#     print(img.shape)


# path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d\HR\50525.mat'
# file = io.loadmat(path)
# img = file['f1']
# print(img.shape)
# conv=common.default_conv
# m.append(conv(n_feats, 8 * n_feats, 3, bias))
# m.append(nn.PixelShuffle3d(2))