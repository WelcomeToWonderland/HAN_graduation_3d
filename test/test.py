import cv2
import numpy as np
import argparse
import os
import datetime
import torch


temp = np.array([1, 2, 1, 4])
temp = np.where(temp == 1, 2, temp)
print(temp)
