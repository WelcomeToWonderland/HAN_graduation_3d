# —*- coding = utf-8 -*-
# @Time : 2023-03-02 21:11
# @Author : 阙祥辉
# @File : psnr_ssim.py
# @Software : PyCharm

import os
import cv2
# from skimage.metrics import mean_squared_error
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.tensorboard import SummaryWriter
import torch
import datetime
import numpy as np


def psnr_ssim_img():
    dataset = "Manga109"
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_dir = os.path.join('.', f"{dataset}_{now}")
    writer = SummaryWriter(log_dir=log_dir)
    log_file = open(log_dir + r"/log.txt", 'x')

    path_sr = r'/root/autodl-tmp/project/HAN/experiment/HANx2_test_noshiftmean_Manga109/results-Manga109'
    path_hr = r'/root/autodl-tmp/dataset/Manga109/HR'
    img_sr_list = os.listdir(path_sr)
    img_hr_list = os.listdir(path_hr)
    img_sr_list.sort()
    img_hr_list.sort()

    num = 0
    psnr_mean = 0
    ssim_mean = 0
    for img_sr, img_hr in zip(img_sr_list, img_hr_list):
        num += 1

        path_sr_ = os.path.join(path_sr, img_sr)
        path_hr_ = os.path.join(path_hr, img_hr)
        img_sr_ = cv2.imread(path_sr_)
        img_hr_ = cv2.imread(path_hr_)

        print(f"shape_hr:{np.shape(img_hr_)}, shape_sr:{np.shape(img_sr_)}")

        psnr = peak_signal_noise_ratio(img_hr_, img_sr_, data_range=255)
        ssim = structural_similarity(img_hr_, img_sr_, multichannel=True)

        psnr_mean += psnr
        ssim_mean += ssim

        log = f"ordinal:{num}, img_sr:{img_sr}, img_hr:{img_hr} : psnr:{psnr.item()}, ssim:{ssim.item()}" + '\n'
        log_file.write(log)
        print(log)

        writer.add_scalar(r'psnr', psnr.item(), num)
        writer.add_scalar(r'ssim', ssim.item(), num)

    psnr_mean /= num
    ssim_mean /= num
    log = f"psnr_mean:{psnr_mean.item()}, ssim_mean:{ssim_mean.item()}"
    log_file.write(log)
    print(log)
    log_file.close();

def psnr_ssim_dat(dataset, path_sr, path_hr):
    dataset = "Manga109"
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_dir = os.path.join('.', f"{dataset}_{now}")
    writer = SummaryWriter(log_dir=log_dir)
    log_file = open(log_dir + r"/log.txt", 'x')

    img_sr_list = os.listdir(path_sr)
    img_hr_list = os.listdir(path_hr)
    img_sr_list.sort()
    img_hr_list.sort()

    # supported_img_formats = (".DAT")

    num = 0
    psnr_mean = 0
    ssim_mean = 0
    for img_sr, img_hr in zip(img_sr_list, img_hr_list):
        # if not img_sr.endswith(supported_img_formats):
        #     continue
        # if not img_hr.endswith(supported_img_formats):
        #     continue

        num += 1

        path_sr_ = os.path.join(path_sr, img_sr)
        path_hr_ = os.path.join(path_hr, img_hr)
        img_sr_ = cv2.imread(path_sr_)
        img_hr_ = cv2.imread(path_hr_)

        print(f'\nimg_sr:{img_sr}, img_hr;{img_hr}')
        print(f"\nshape_hr:{np.shape(img_hr_)}, shape_sr:{np.shape(img_sr_)}")

        psnr = peak_signal_noise_ratio(img_hr_, img_sr_, data_range=255)
        ssim = structural_similarity(img_hr_, img_sr_, multichannel=True)

        psnr_mean += psnr
        ssim_mean += ssim

        log = f"ordinal:{num}, img_sr:{img_sr}, img_hr:{img_hr} : psnr:{psnr.item()}, ssim:{ssim.item()}" + '\n'
        log_file.write(log)
        print(log)

        writer.add_scalar(r'psnr', psnr.item(), num)
        writer.add_scalar(r'ssim', ssim.item(), num)

    psnr_mean /= num
    ssim_mean /= num
    log = f"psnr_mean:{psnr_mean.item()}, ssim_mean:{ssim_mean.item()}"
    log_file.write(log)
    print(log)
    log_file.close();



if __name__ == '__main__':
    dataset = f'Neg_35_Left_test'
    path_sr = f'/root/autodl-tmp/project/HAN_for_test/experiment/HANx2_testForRunning_oabreast/results-Neg_35_Left_test/'
    path_hr = f'/root/autodl-tmp/dataset/OABreast/Neg_35_Left_test/HR/'
    dimensions = []
    psnr_ssim_dat(dataset = dataset, path_sr = path_sr, path_hr = path_hr)