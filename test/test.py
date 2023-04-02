# —*- coding = utf-8 -*-
# @Time : 2023-03-03 15:48
# @Author : 阙祥辉
# @File : test.py
# @Software : PyCharm

import cv2
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--nx', type=int)
parser.add_argument('--ny', type=int)
parser.add_argument('--nz', type=int)
args = parser.parse_args()

t1 = np.zeros((3,3,3), dtype=np.uint8)

t2 = t1[0:2, 0:2, :]

print(np.shape(t2))






