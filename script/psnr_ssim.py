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
parser.add_argument('--nx', type=int)
parser.add_argument('--ny', type=int)
parser.add_argument('--nz', type=int)
args = parser.parse_args()

def psnr_ssim_img():
    print('\npsnr_ssim_img')
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
    label = f"psnr_{dataset}"
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.plot(axis, psnr)
    plt.xlabel = 'idx'
    plt.ylabel = 'psnr'
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{label}.png"))
    plt.close(fig)

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
    loss_function = nn.MSELoss()

    print(args.sr_path)
    print(args.hr_path)
    sr_dat = np.fromfile(args.sr_path, dtype=np.uint8)
    sr_dat = sr_dat.reshape(args.nx, args.ny, args.nz)
    hr_dat = np.fromfile(args.hr_path, dtype=np.uint8)
    hr_dat = hr_dat.reshape(args.nx, args.ny, args.nz)


    psnr = []
    ssim = []
    loss = []
    psnr_mean = 0.0
    ssim_mean = 0.0
    loss_mean = 0.0
    for idx in range(args.nz):
        psnr_temp = peak_signal_noise_ratio(hr_dat[:, :, idx], sr_dat[:, :, idx], data_range=4)
        ssim_temp = structural_similarity(hr_dat[:, :, idx], sr_dat[:, :, idx], data_range=4, multichannel=False)
        loss_temp = loss_function(torch.from_numpy(hr_dat[:, :, idx]).float(), torch.from_numpy(sr_dat[:, :, idx]).float())
        # print(f"1 : {ssim_temp}")
        # ssim_temp = structural_similarity(hr_dat[:, :, idx], sr_dat[:, :, idx])
        # print(f"2 : {ssim_temp}")
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        loss_mean += loss_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        loss.append(loss_temp)
        # log = f"ordinal:{idx+1} : psnr:{psnr.item()}, ssim:{ssim.item()}" + '\n'
        log = f"\nordinal:{idx+1} : psnr:{psnr_temp}, ssim:{ssim_temp}, loss:{loss_temp}"
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx+1)
        writer.add_scalar(r'ssim', ssim_temp, idx+1)
        writer.add_scalar(r'loss', loss_temp, idx+1)

    psnr_mean /= args.nz
    ssim_mean /= args.nz
    loss_mean /= args.nz
    log = f"\npsnr_mean : {psnr_mean}, ssim_mean : {ssim_mean}, loss_mean : {loss_mean}"
    log_file.write(log)
    print(log)
    psnr_whole = peak_signal_noise_ratio(hr_dat, sr_dat, data_range=4)
    ssim_whole = structural_similarity(hr_dat, sr_dat, data_range=4, multichannel=True)
    loss_whole = loss_function(torch.form_numpy(hr_dat).float(), torch.from_numpy(sr_dat).float())
    log = f"\nthe whole : psnr:{psnr_whole}, ssim:{ssim_whole}, loss:{loss_whole}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, args.nz, args.nz)
    label = f"psnr_{dataset}"
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.plot(axis, psnr)
    plt.xlabel = 'idx'
    plt.ylabel = 'psnr'
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{label}.png"))
    plt.close(fig)

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

    label = f"loss_{dataset}"
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, loss, label=label)
    plt.legend()
    plt.xlabel = 'idx'
    plt.ylabel = 'loss'
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{label}.png"))
    plt.close(fig)

def psnr_ssim_dat_3d():
    print('\npsnr_ssim_dat_3d')
    nxs = [616, 284, 494]
    nys = [484, 410, 614]
    nzs = [718, 722, 752]
    dataset = args.dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # tb
    log_dir = os.path.join('.', 'psnr_ssim_logs', f"{dataset}_{now}")
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
    psnr = []
    ssim = []
    loss = []
    psnr_mean = 0.0
    ssim_mean = 0.0
    loss_mean = 0.0
    supported_formats = ('.DAT')
    length = len(hr_list)
    for idx_filename in range(length):
        hr_filename = hr_list[idx_filename]
        sr_filename = sr_list[idx_filename]
        if hr_filename != sr_filename or not hr_filename.endswith(supported_formats):
            """
            可能存在情况：一对数据错位，后面的所有数据都无法处理
            """
            continue
        # 确定三维
        if hr_filename.split('_')[1] == '07':
            idx = 0
        elif hr_filename.split('_')[1] == '35':
            idx = 1
        elif hr_filename.split('_')[1] == '47':
            idx = 2
        nx = nxs[idx]
        ny = nys[idx]
        nz = nzs[idx]
        # 读取文件
        hr_dat = np.fromfile(os.path.join(hr_dir, hr_filename), dtype=np.uint8)
        hr_dat = hr_dat.reshape(nx, ny, nz)
        sr_dat = np.fromfile(os.path.join(sr_dir, sr_filename), dtype=np.uint8)
        sr_dat = sr_dat.reshape(nx, ny, nz)
        # 计算
        psnr_temp = peak_signal_noise_ratio(hr_dat, sr_dat, data_range=4)
        ssim_temp = structural_similarity(hr_dat, sr_dat, data_range=4, multichannel=True)
        loss_temp = loss_function(torch.from_numpy(hr_dat).float(), torch.from_numpy(sr_dat).float())
        psnr_mean += psnr_temp
        ssim_mean += ssim_temp
        loss_mean += loss_temp
        psnr.append(psnr_temp)
        ssim.append(ssim_temp)
        loss.append(loss_temp)
        log = f"\nordinal:{idx+1} : psnr:{psnr_temp}, ssim:{ssim_temp}, loss:{loss_temp}"
        log_file.write(log)
        print(log)
        writer.add_scalar(r'psnr', psnr_temp, idx_filename+1)
        writer.add_scalar(r'ssim', ssim_temp, idx_filename+1)
        writer.add_scalar(r'loss', loss_temp, idx_filename+1)

    psnr_mean /= length
    ssim_mean /= length
    loss_mean /= length
    log = f"\npsnr_mean : {psnr_mean}, ssim_mean : {ssim_mean}, loss_mean : {loss_mean}"
    log_file.write(log)
    print(log)
    log_file.close();

    axis = np.linspace(1, length, length)
    label = f"psnr_{dataset}"
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, psnr, label=label)
    plt.legend()
    plt.plot(axis, psnr)
    plt.xlabel = 'idx'
    plt.ylabel = 'psnr'
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{label}.png"))
    plt.close(fig)

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

    label = f"loss_{dataset}"
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, loss, label=label)
    plt.legend()
    plt.xlabel = 'idx'
    plt.ylabel = 'loss'
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"{label}.png"))
    plt.close(fig)

if __name__ == '__main__':
    args.dataset = 'OABreast_3d'
    args.data_dir = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\3D"
    psnr_ssim_dat_3d()

    # args.dataset = r'Neg_07_Left_test'
    # # args.hr_path = r'/root/autodl-tmp/dataset/OABreast/downing/Neg_07_Left_test/HR/MergedPhantom.DAT'
    # # args.sr_path = r'/root/autodl-tmp/project/HAN_for_test/experiment/2023-04-06-19:53:58HANx2_oabreast/results-Neg_07_Left_test/MergedPhantom_x2_SR.DAT'
    # args.hr_path = r'D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_07_Left_test\HR\MergedPhantom.DAT'
    # args.sr_path = r'D:\workspace\HAN_for_test\experiment\2023-04-06-19%3A53%3A58HANx2_oabreast\results-Neg_07_Left_test\MergedPhantom_x2_SR.DAT'
    # nxs = [616, 284, 494]
    # nys = [484, 410, 614]
    # """
    # original
    # train
    # test
    # """
    # nzs = [719, 722, 752,
    #        319, 322, 352,
    #        400, 400, 400]
    # if args.dataset.split('_')[1] == '07':
    #     idx = 0
    # elif args.dataset.split('_')[1] == '35':
    #     idx = 1
    # elif args.dataset.split('_')[1] == '47':
    #     idx = 2
    # if args.dataset.endswith('train'):
    #     multiple = 1
    # elif args.dataset.endswith('test'):
    #     multiple = 2
    # else:
    #     multiple = 0
    # args.nx = nxs[idx]
    # args.ny = nys[idx]
    # args.nz = nzs[3 * multiple + idx]
    # psnr_ssim_dat()


    # 所有dat文件
    # d1 = 'OABreast_Neg_'
    # d2 = '_Left'
    # h1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    # h2 = r"_Left\HR\MergedPhantom.DAT"
    # s1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    # s2 = r"_Left\SR\X2\MergedPhantom.DAT"
    # datasets = ['07', '35', '47']
    # nxs = [616, 284, 494]
    # nys = [484, 410, 614]
    # nzs = [719, 722, 752]
    # for idx in range(3):
    #     args.dataset = d1 + datasets[idx] + d2
    #     args.hr_path = h1 + datasets[idx] + h2
    #     args.sr_path = s1 + datasets[idx] + s2
    #     args.nx = nxs[idx]
    #     args.ny = nys[idx]
    #     args.nz = nzs[idx]
    #     psnr_ssim_dat()

    # png图片
    # args.dataset = 'Manga109'
    # args.hr_path = 'D:\workspace\dataset\Manga109\clipping\HR'
    # args.sr_path = 'D:\workspace\dataset\Manga109\clipping\SR\X2'
    # psnr_ssim_img()