import cv2
import numpy as np
import argparse
import os
import datetime
import torch
import re
from scipy import io

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

path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d\HR'
for filename in os.listdir(path):
    file = io.loadmat(os.path.join(path, filename))
    data = file['f1']
    print(f"\nfilename : {filename}")
    print(f"shape : {data.shape}")