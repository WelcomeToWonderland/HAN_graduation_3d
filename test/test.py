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


def augment(data):
    if random.random() < 0.5:
        data = data[::-1]
    if random.random() < 0.5:
        data = data[:, ::-1]

    if np.ndim(data)==3 and random.random() < 0.5:
        data = data[:, :, ::-1]
    if random.random() < 0.5:
        angle = random.uniform(0, 360)
        data = rotate(data, angle=angle, reshape=False, mode='nearest')
    # if random.random() < 0.5:
    #     angle = random.uniform(0, 360)
    #     data = rotate(data, angle=angle, axes=1, reshape=True, mode='nearest')
    # if random.random() < 0.5:
    #     angle = random.uniform(0, 360)
    #     data = rotate(data, angle=angle, axes=2, reshape=True, mode='nearest')
    return data

# data = np.ndarray(shape=(4, 5, 6))
# for idx in range(10):
#     print(augment(data).shape)

data = np.ndarray(shape=(9, 10))
for idx in range(10):
    print(augment(data).shape)
