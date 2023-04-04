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

# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--hr_path', type=str, default=r'',
                    help='path to high resolution image dir')
parser.add_argument('--sr_path', type=str, default=r'',
                    help='path to super resolution image dir')
parser.add_argument('--dataset', type=str, default=r'',
                    help='')
parser.add_argument('--nx', type=int)
parser.add_argument('--ny', type=int)
parser.add_argument('--nz', type=int)
args = parser.parse_args()

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

def psnr_ssim_dat():
    print('\npsnr_ssim_dat')
    dataset = args.dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # tb
    log_dir = os.path.join('.', 'psnr_ssim_logs', f"{dataset}_{now}")
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # log
    log_file = open(log_dir + r"/log.txt", 'x')

    print(args.sr_path)
    print(args.hr_path)
    sr_dat = np.fromfile(args.sr_path, dtype=np.uint8)
    sr_dat = sr_dat.reshape(args.nx, args.ny, args.nz)
    hr_dat = np.fromfile(args.hr_path, dtype=np.uint8)
    hr_dat = hr_dat.reshape(args.nx, args.ny, args.nz)


    psnr = []
    ssim = []
    psnr_mean = 0
    ssim_mean = 0
    for idx in range(args.nz):
        psnr_temp = peak_signal_noise_ratio(hr_dat[:, :, idx], sr_dat[:, :, idx], data_range=5)
        ssim_temp = structural_similarity(hr_dat[:, :, idx], sr_dat[:, :, idx], multichannel=False)
        # print(f"1 : {ssim_temp}")
        # ssim_temp = structural_similarity(hr_dat[:, :, idx], sr_dat[:, :, idx])
        # print(f"2 : {ssim_temp}")
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        # log = f"ordinal:{idx+1} : psnr:{psnr.item()}, ssim:{ssim.item()}" + '\n'
        log = f"ordinal:{idx+1} : psnr:{psnr_temp}, ssim:{ssim_temp}"
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx+1)
        writer.add_scalar(r'ssim', ssim_temp, idx+1)

    psnr_mean /= args.nz
    ssim_mean /= args.nz
    log = f"\npsnr_mean : {psnr_mean}, ssim_mean : {ssim_mean}"
    log_file.write(log)
    print(log)
    psnr_whole = peak_signal_noise_ratio(hr_dat, sr_dat, data_range=5)
    ssim_whole = structural_similarity(hr_dat, sr_dat, multichannel=True)
    log = f"\nthe whole : psnr:{psnr_whole}, ssim:{ssim_whole}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, args.nz, args.nz)
    label = f"psnr_{dataset}"
    fig = plt.figure()
    plt.title(label)
    # plt.plot(axis, psnr, label=label)
    # plt.legend()
    plt.plot(axis, psnr)
    plt.xlabel = 'idx'
    plt.ylabel = 'psnr'
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{label}.png"))
    plt.close(fig)

    # label = 'ssim'
    label = f"ssim_{dataset}"
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, ssim, label=label)
    plt.legend()
    plt.xlabel = 'idx'
    plt.ylabel = 'ssim'
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{label}.png"))
    plt.close(fig)


if __name__ == '__main__':
    psnr_ssim_dat()