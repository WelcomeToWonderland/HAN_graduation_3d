import os
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.tensorboard import SummaryWriter
import torch
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from src.utility import get_3d
from scipy import io
from script.common import delete_folder

def remove_zero(dir_original, dir_processed):
    # 创建存储文件夹
    # delete_folder(dir_processed)
    os.makedirs(dir_processed, exist_ok=True)
    # 遍历文件夹
    for filename in os.listdir(dir_original):
        # 加载数据
        file = io.loadmat(io.loadmat(os.path.join(dir_original, filename)))
        data = file['img']
        # 剔除元素全为零的切片
        print(f"before : {data.shape}")
        _, _, length = data.shape
        result = None
        for iz in range(length):
            temp = data[..., iz]
            if not np.all(temp == 0):
                if not result:
                    result = temp
                else:
                    result = np.concatenate(result, temp)
        print(f"after : {result.shape}")
        # 存储经过处理的文件
        path_save = os.path.join(dir_processed, filename)
        file['img'] = result
        io.savemat(path_save, file)
    None

if __name__ == '__main__':
    path =