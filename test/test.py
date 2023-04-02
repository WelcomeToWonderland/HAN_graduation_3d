import cv2
import numpy as np
import argparse
import os
import datetime
import torch


temp = np.zeros((1000, 1000, 1000))
print(f"dtype : {temp.dtype}")
temp = cv2.resize(temp, (2000, 2000), interpolation = cv2.INTER_CUBIC)
print(temp.shape)

