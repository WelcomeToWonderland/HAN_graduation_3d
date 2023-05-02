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

# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--data_dir', type=str, default=r'',
                    help='')
parser.add_argument('--hr_path', type=str, default=r'',
                    help='path to high resolution image dir')
parser.add_argument('--sr_path', type=str, default=r'',
                    help='path to super resolution image dir')
parser.add_argument('--dataset', type=str, default=r'',
                    help='')
parser.add_argument('--algorithm', type=str, default=r'',
                    help='')
parser.add_argument('--pixel_range', type=int, default=255,
                    help='')
args = parser.parse_args()

def psnr_ssim_img():
    print('\npsnr_ssim_img')
    dataset = args.dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # tb
    log_dir = os.path.join('../script_results', '../script_results/psnr_ssim_logs', f"{args.algorithm}_{dataset}_{now}")
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # log
    log_file = open(log_dir + r"/log.txt", 'x')

    print(args.sr_path)
    print(args.hr_path)
    path_sr = args.sr_path
    path_hr = args.hr_path
    img_sr_list = os.listdir(path_sr)
    img_hr_list = os.listdir(path_hr)
    img_sr_list.sort()
    img_hr_list.sort()

    psnr = []
    ssim = []
    psnr_mean = 0
    ssim_mean = 0
    for idx in range(len(img_hr_list)):
        path_sr_ = os.path.join(path_sr, img_sr_list[idx])
        path_hr_ = os.path.join(path_hr, img_hr_list[idx])
        img_sr_ = cv2.imread(path_sr_)
        img_hr_ = cv2.imread(path_hr_)
        psnr_temp = peak_signal_noise_ratio(img_hr_, img_sr_, data_range=255)
        ssim_temp = structural_similarity(img_hr_, img_sr_, multichannel=True)
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        log = f"ordinal:{idx+1} : psnr:{psnr_temp}, ssim:{ssim_temp}" + '\n'
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx+1)
        writer.add_scalar(r'ssim', ssim_temp, idx+1)

    psnr_mean /= len(img_hr_list)
    ssim_mean /= len(img_hr_list)
    log = f"psnr_mean:{psnr_mean}, ssim_mean:{ssim_mean}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, len(img_hr_list), len(img_hr_list))
    label = f"{dataset}"
    title = args.algorithm + '_psnr'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

    label = f"{dataset}"
    title = args.algorithm + '_ssim'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, ssim, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

def psnr_ssim_dat():
    """
    输入文件路径
    :return:
    """
    print('\npsnr_ssim_dat')
    dataset = args.dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # tb
    log_dir = os.path.join('.', '../script_results/psnr_ssim_logs', f"{dataset}_{now}")
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # log
    log_file = open(log_dir + r"/log.txt", 'x')

    print(args.sr_path)
    print(args.hr_path)
    # idx = dataset.index('_')
    # basename = dataset[idx+1:]
    nx, ny, nz = get_3d(dataset)
    sr_dat = np.fromfile(args.sr_path, dtype=np.uint8)
    sr_dat = sr_dat.reshape(nx, ny, nz)
    hr_dat = np.fromfile(args.hr_path, dtype=np.uint8)
    hr_dat = hr_dat.reshape(nx, ny, nz)


    psnr = []
    ssim = []
    psnr_mean = 0.0
    ssim_mean = 0.0
    for idx in range(nz):
        psnr_temp = peak_signal_noise_ratio(hr_dat[:, :, idx], sr_dat[:, :, idx], data_range=4)
        ssim_temp = structural_similarity(hr_dat[:, :, idx], sr_dat[:, :, idx], data_range=4, multichannel=False)
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        log = f"\nordinal:{idx+1} : psnr:{psnr_temp}, ssim:{ssim_temp}"
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx+1)
        writer.add_scalar(r'ssim', ssim_temp, idx+1)

    psnr_mean /= nz
    ssim_mean /= nz
    log = f"\npsnr_mean : {psnr_mean}, ssim_mean : {ssim_mean}"
    log_file.write(log)
    print(log)
    psnr_whole = peak_signal_noise_ratio(hr_dat, sr_dat, data_range=4)
    ssim_whole = structural_similarity(hr_dat, sr_dat, data_range=4, multichannel=True)
    log = f"\nthe whole : psnr:{psnr_whole}, ssim:{ssim_whole}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, nz, nz)
    label = f"{dataset}"
    title = args.algorithm + '_psnr'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

    label = f"{dataset}"
    title = args.algorithm + '_ssim'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, ssim, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

def psnr_ssim_dat_3d():
    """
    输入文件夹路径
    :return:
    """
    print('\npsnr_ssim_dat_3d')
    dataset = args.dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # tb
    log_dir = os.path.join('.', '../script_results/psnr_ssim_logs', f"{dataset}_{now}")
    print(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    # log
    log_file = open(log_dir + r"/log.txt", 'x')
    loss_function = nn.MSELoss()

    hr_dir = os.path.join(args.data_dir, 'HR')
    sr_dir = os.path.join(args.data_dir, 'SR', 'X2')
    print(f"hr_dir : {hr_dir}")
    print(f"sr_dir : {sr_dir}")
    hr_list = sorted(os.listdir(hr_dir))
    sr_list = sorted(os.listdir(sr_dir))
    supported_formats = ('.DAT')
    length = len(hr_list)
    psnr_mean = 0.0
    ssim_mean = 0.0
    psnr = []
    ssim = []
    for idx_filename in range(length):
        hr_filename = hr_list[idx_filename]
        sr_filename = sr_list[idx_filename]
        if hr_filename != sr_filename or not hr_filename.endswith(supported_formats):
            """
            可能存在情况：一对数据错位，后面的所有数据都无法处理
            """
            continue
        # 确定三维
        nx, ny, nz = get_3d(os.path.splitext(hr_filename)[0])
        # 读取文件
        hr_dat = np.fromfile(os.path.join(hr_dir, hr_filename), dtype=np.uint8)
        hr_dat = hr_dat.reshape(nx, ny, nz)
        sr_dat = np.fromfile(os.path.join(sr_dir, sr_filename), dtype=np.uint8)
        sr_dat = sr_dat.reshape(nx, ny, nz)
        # 计算
        psnr_temp = peak_signal_noise_ratio(hr_dat, sr_dat, data_range=4)
        ssim_temp = structural_similarity(hr_dat, sr_dat, data_range=4, multichannel=True)
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        log = f"\nordinal:{idx_filename+1} : psnr:{psnr_temp}, ssim:{ssim_temp}"
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx_filename+1)
        writer.add_scalar(r'ssim', ssim_temp, idx_filename+1)

    psnr_mean /= length
    ssim_mean /= length
    log = f"\npsnr_mean : {psnr_mean}, ssim_mean : {ssim_mean}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, length, length)
    label = f"{dataset}"
    title = args.algorithm + '_psnr'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

    label = f"{dataset}"
    title = args.algorithm + '_ssim'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, ssim, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

def psnr_ssim_mat():
    print('\npsnr_ssim_mat')
    dataset = args.dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # make dirs
    log_dir = os.path.join('.', '../script_results/psnr_ssim_logs', f"{dataset}_{now}")
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    # tb
    writer = SummaryWriter(log_dir=log_dir)
    # log
    log_file = open(log_dir + r"/log.txt", 'x')

    print(args.sr_path)
    print(args.hr_path)

    data = io.loadmat(args.sr_path)
    sr_mat = data['f1']
    data = io.loadmat(args.hr_path)
    hr_mat = data['f1']


    psnr = []
    ssim = []
    psnr_mean = 0.0
    ssim_mean = 0.0
    for idx in range(hr_mat.shape[2]):
        psnr_temp = peak_signal_noise_ratio(hr_mat[:, :, idx], sr_mat[:, :, idx], data_range=1.0e3)
        ssim_temp = structural_similarity(hr_mat[:, :, idx], sr_mat[:, :, idx], data_range=1.0e3, multichannel=False)
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        log = f"\nordinal:{idx+1} : psnr:{psnr_temp}, ssim:{ssim_temp}"
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx+1)
        writer.add_scalar(r'ssim', ssim_temp, idx+1)

    psnr_mean /= sr_mat.shape[2]
    ssim_mean /= sr_mat.shape[2]
    log = f"\npsnr_mean : {psnr_mean}, ssim_mean : {ssim_mean}"
    log_file.write(log)
    print(log)
    psnr_whole = peak_signal_noise_ratio(hr_mat, sr_mat, data_range=1.0e3)
    ssim_whole = structural_similarity(hr_mat, sr_mat, data_range=1.0e3, multichannel=True)
    log = f"\nthe whole : psnr:{psnr_whole}, ssim:{ssim_whole}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, sr_mat.shape[2], sr_mat.shape[2])
    label = f"{dataset}"
    title = args.algorithm + '_PSNR'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

    label = f"{dataset}"
    title = args.algorithm + '_SSIM'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, ssim, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

def psnr_ssim_mat_3d():
    """
    输入文件夹路径
    :return:
    """
    print('\npsnr_ssim_mat_3d')
    dataset = args.dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # make dirs
    log_dir = os.path.join('.', '../script_results/psnr_ssim_logs', f"{args.algorithm}_{dataset}_{now}")
    os.makedirs(log_dir, exist_ok=True)
    print(log_dir)
    # tb
    writer = SummaryWriter(log_dir=log_dir)
    # log
    log_file = open(log_dir + r"/log.txt", 'x')

    hr_dir = args.hr_path
    sr_dir = args.sr_path

    # hr_dir = os.path.join(args.data_dir, 'HR')
    # sr_dir = os.path.join(args.data_dir, 'SR', 'X2')
    print(f"hr_dir : {hr_dir}")
    print(f"sr_dir : {sr_dir}")
    hr_list = sorted(os.listdir(hr_dir))
    sr_list = sorted(os.listdir(sr_dir))
    supported_formats = ('.mat')
    length = len(hr_list)
    psnr_mean = 0.0
    ssim_mean = 0.0
    psnr = []
    ssim = []
    for idx_filename in range(length):
        hr_filename = hr_list[idx_filename]
        sr_filename = sr_list[idx_filename]
        # if hr_filename != sr_filename or not hr_filename.endswith(supported_formats):
        #     """
        #     可能存在情况：一对数据错位，后面的所有数据都无法处理
        #     """
        #     continue
        # 读取文件
        file1 = io.loadmat(os.path.join(hr_dir, hr_filename))
        hr_mat = file1['f1']
        file2 = io.loadmat(os.path.join(sr_dir, sr_filename))
        sr_mat = file2['f1']
        # 计算
        print(f"hr shape : {hr_mat.shape}")
        print(f"sr shape : {sr_mat.shape}")
        psnr_temp = peak_signal_noise_ratio(hr_mat, sr_mat, data_range=args.pixel_range)
        ssim_temp = structural_similarity(hr_mat, sr_mat, data_range=args.pixel_range, multichannel=True)
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        log = f"\nordinal:{idx_filename+1} : psnr:{psnr_temp}, ssim:{ssim_temp}"
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx_filename+1)
        writer.add_scalar(r'ssim', ssim_temp, idx_filename+1)

    psnr_mean /= length
    ssim_mean /= length
    log = f"\npsnr_mean : {psnr_mean}, ssim_mean : {ssim_mean}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, length, length)
    label = f"{dataset}"
    title = args.algorithm + '_PSNR'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('PSNR')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)

    label = f"{dataset}"
    title = args.algorithm + '_SSIM'
    fig = plt.figure()
    plt.title(title)
    plt.plot(axis, ssim, label=label)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('SSIM')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{title}.png"))
    plt.close(fig)


if __name__ == '__main__':
    args.algorithm = 'bicubic'

    # png图片
    # datasets = ['Set5', 'Set14', 'Urban100', 'Manga109']
    # prefix_hr = r'/root/autodl-tmp/dataset'
    # prefix_sr = r'/root/autodl-tmp/project/HAN_for_3d/experiment/div2k_urban100_bn_best/results-'
    # suffix_hr = r'HR'
    # for dataset in datasets:
    #     args.dataset = dataset
    #     args.hr_path = os.path.join(prefix_hr, dataset, suffix_hr)
    #     args.sr_path = prefix_sr + dataset
    #     psnr_ssim_img()

    # datasets = ['Set5', 'Set14', 'Urban100', 'Manga109']
    # prefix_hr = r'D:\workspace\dataset'
    # prefix_sr = r'D:\workspace\dataset'
    # suffix_hr = r'HR'
    # suffix_sr = r'SR/X2'
    # for dataset in datasets:
    #     args.dataset = dataset
    #     args.hr_path = os.path.join(prefix_hr, dataset, suffix_hr)
    #     args.sr_path = os.path.join(prefix_sr, dataset, suffix_sr)
    #     psnr_ssim_img()

    # args.dataset = 'Urban100'
    # args.hr_path = r'D:\workspace\dataset\Urban100\HR'
    # args.sr_path = r'D:\workspace\dataset\Urban100\SR\X2'
    # psnr_ssim_img()

    # # args.dataset = r'Neg_07_Left_test'
    # # # args.hr_path = r'/root/autodl-tmp/dataset/OABreast/downing/Neg_07_Left_test/HR/MergedPhantom.DAT'
    # # # args.sr_path = r'/root/autodl-tmp/project/HAN_for_test/experiment/2023-04-06-19:53:58HANx2_oabreast/results-Neg_07_Left_test/MergedPhantom_x2_SR.DAT'
    # # args.hr_path = r'D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_07_Left_test\HR\MergedPhantom.DAT'
    # # args.sr_path = r'D:\workspace\HAN_for_test\experiment\2023-04-06-19%3A53%3A58HANx2_oabreast\results-Neg_07_Left_test\MergedPhantom_x2_SR.DAT'
    # # psnr_ssim_dat()
    #
    # # 2d 所有dat文件
    # d1 = 'OABreast_Neg_'
    # d2 = '_Left'
    # h1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    # h2 = r"_Left\HR\MergedPhantom.DAT"
    # s1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    # s2 = r"_Left\SR\X2\MergedPhantom.DAT"
    # datasets = ['07', '35', '47']
    # for idx in range(3):
    #     args.dataset = d1 + datasets[idx] + d2
    #     args.hr_path = h1 + datasets[idx] + h2
    #     args.sr_path = s1 + datasets[idx] + s2
    #     psnr_ssim_dat()

    # args.dataset = r'Neg_07_Left_test'
    # args.hr_path = r'/root/autodl-tmp/dataset/OABreast_2d/Neg_07_Left_test/HR/MergedPhantom.DAT'
    # args.sr_path = r'/root/autodl-tmp/project/HAN_for_3d/experiment/oabreast_2d_transfer_learning/results-Neg_07_Left_test/Neg_07_Left_test_x2_SR.DAT'
    # psnr_ssim_dat()

    #
    # # 3d dat
    # args.dataset = 'OABreast_3d'
    # args.data_dir = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\3D"
    # psnr_ssim_dat_3d()
    #
    # mat 2d 输入文件路径
    # path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_2d_uint'
    # for foldername in os.listdir(path):
    #     args.dataset = 'usct' + '_' + foldername
    #     args.hr_path = os.path.join(path, foldername, 'HR', foldername+'.mat')
    #     args.sr_path = os.path.join(path, foldername, 'SR', 'X2', foldername+'.mat')
    #     psnr_ssim_mat()

    # rm -rf /root/autodl-tmp/project/HAN_for_3d/script/psnr_ssim_logs/*
    # datasets = ['20220511T153240', '20220517T112745', '50525']
    # prefix_sr = r'/root/autodl-tmp/project/HAN_for_3d/experiment/HANx2_usct_2d_bn_other_lr_3'
    # prefix_hr = r'/root/autodl-tmp/dataset/USCT_2d/every_other_points_2d_float'
    # for dataset in datasets:
    #     args.dataset = 'usct' + '_' + dataset
    #     args.sr_path = os.path.join(prefix_sr, 'results-'+dataset, dataset+'_x2_SR.mat')
    #     args.hr_path = os.path.join(prefix_hr, dataset, 'HR', dataset+'.mat')
    #     psnr_ssim_mat()
    #
    # # mat 3d 输入HR和SR文件夹路径
    # args.dataset = 'usct' + '_' + '3d'
    # args.data_dir = r"/root/autodl-tmp/dataset/USCT_3d/every_other_points_2d_float/"
    # psnr_ssim_mat_3d()

    args.pixel_range = 4
    args.dataset = 'usct_3d_to_oabreast'
    args.sr_path = r'/workspace/projects/HAN_3d_53408/experiment/3d_to_oabreast/SR/X2/'
    args.hr_path = r'/workspace/projects/HAN_3d_53408/experiment/3d_to_oabreast/HR/'
    psnr_ssim_mat_3d()
