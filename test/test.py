import cv2
import numpy as np
import argparse
import os
import datetime
import torch
import cv2

path = r'D:\workspace\dataset\DIV2K\DIV2K_train_HR\0001.png'
img = cv2.imread(path)
print(img.dtype)

