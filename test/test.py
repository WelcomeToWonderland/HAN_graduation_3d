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

figures = ['a', 'b', 'c']
temp = 'test'
for figure in figures:
    temp = os.path.join(temp, figure)
    print(temp)
print(temp)