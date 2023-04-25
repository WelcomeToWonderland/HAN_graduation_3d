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

path = r'D:\workspace\dataset\USCT\original\HR\50525.mat'
file = io.loadmat(path)
data = file['f1']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = np.indices(data.shape)
ax.scatter(x, y, z, c=data.ravel(), cmap=plt.cm.viridis)
plt.show()






































# # 精简
# print(f"\n2d")
# path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\2d\20221116T164200\HR\20221116T164200.mat'
# file = io.loadmat(path)
# hr = file['f1']
# shape = hr.shape
# lr = np.zeros((shape[0]//2, shape[1]//2, shape[2]))
# for idx in range(shape[2]):
#     lr[..., idx] = cv2.resize(hr[..., idx], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# lr = np.clip(0.0, 1.0e3, lr)
# sr = np.zeros(hr.shape)
# for idx in range(hr.shape[2]):
#     sr[..., idx] = cv2.resize(lr[..., idx], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# sr = np.clip(0.0, 1.0e3, sr)
# psnr = peak_signal_noise_ratio(hr, sr, data_range=1.0e3)
# ssim = structural_similarity(hr, sr, data_range=1.0e3, multichannel=True)
# print(f"psnr : {psnr}")
# print(f"ssim : {ssim}")
#
# print(f"\n3d")
# path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\2d\20221116T164200\HR\20221116T164200.mat'
# file = io.loadmat(path)
# hr = file['f1']
# lr = zoom(hr, (0.5, 0.5, 0.5), order=1)
# lr = np.clip(0.0, 1.0e3, lr)
# sr = zoom(lr, (2, 2, 2), order=1)
# sr = np.clip(0.0, 1.0e3, sr)
# psnr = peak_signal_noise_ratio(hr, sr, data_range=1.0e3)
# ssim = structural_similarity(hr, sr, data_range=1.0e3, multichannel=True)
# print(f"psnr : {psnr}")
# print(f"ssim : {ssim}")