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

# data_1_path = r'D:\workspace\dataset\USCT\clipping\2d\50525\HR\50525.mat'
# print(f"\n{data_1_path}")
# data_1 = io.loadmat(data_1_path)
# print(data_1.keys())
# data_1 = data_1['f1']
# print(type(data_1))
# print(data_1.shape)
# print(f"dtpye : {data_1.dtype}")
# print(f"max : {data_1.max()}")
# print(f"min : {data_1.min()}")
# hist, bins = np.histogram(data_1, np.inf, density=True)
# cumhist = np.cumsum(hist)
# print(f"hist : {hist}")
# print(f"cumhist : {cumhist}")
# print(f"bins : {bins}")
# print('\nflatten')
# hist, bins = np.histogram(data_1.flatten(), np.inf, density=True)
# cumhist = np.cumsum(hist)
# print(f"hist : {hist}")
# print(f"cumhist : {cumhist}")
# print(f"bins : {bins}")


hr = np.random.rand(500, 500, 500) * 1000
print(f"shape : {hr.shape}")
print(f"dtype : {hr.dtype}")
print(f"min : {hr.min()}")
print(f"max : {hr.max()}")
lr = zoom(hr, (0.5, 0.5, 0.5), order=1)
sr = zoom(lr, (2, 2, 2), order=1)
print(f"psnr : {peak_signal_noise_ratio(hr, sr, data_range=1.0e3)}")
print(f"ssim : {structural_similarity(hr, sr, data_range=1.0e3, multichannel=True)}")


