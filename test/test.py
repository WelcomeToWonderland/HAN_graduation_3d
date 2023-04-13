import cv2
import numpy as np
import argparse
import os
import datetime
import torch
import re
from src.utility import get_3d

hr = np.fromfile(r'D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_35_Left\HR\MergedPhantom.DAT', dtype=np.uint8)
nx, ny, nz = get_3d('Neg_35_Left')
hr.reshape(nx, ny, nz)