import cv2
import numpy as np
import argparse
import os
import datetime
import torch
import re
from scipy import io

data_1_path = r'D:\workspace\dataset\USCT\50525lr.mat'
print(f"\n{data_1_path}")
data_1 = io.loadmat(data_1_path)
print(data_1.keys())
data_1 = data_1['imgout']
print(type(data_1))
print(data_1.shape)

data_2_path = r'D:\workspace\dataset\USCT\50525hr.mat'
print(f"\n{data_2_path}")
data_2 = io.loadmat(data_2_path)
print(data_2.keys())
data_2 = data_2['f1']
print(type(data_2))
print(data_2.shape)

data_3_path = r'D:\workspace\dataset\USCT\52748lr.mat'
print(f"\n{data_3_path}")
data_3 = io.loadmat(data_3_path)
print(data_3.keys())
data_3 = data_3['imgout']
print(type(data_3))
print(data_3.shape)

data_4_path = r'D:\workspace\dataset\USCT\52748hr.mat'
print(f"\n{data_4_path}")
data_4 = io.loadmat(data_4_path)
print(data_4.keys())
data_4 = data_4['f1']
print(type(data_4))
print(data_4.shape)