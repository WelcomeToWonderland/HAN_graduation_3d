import os
import argparse
import cv2
import numpy as np
from scipy.ndimage import zoom
from src.utility import get_3d
import re
from scipy import io
import random


# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--dataset', type=str, default=r'',
                    help='')
parser.add_argument('--hr_img_dir', type=str, default=r'D:\workspace\dataset\OABreast\clipping\Neg_35_Left\HR',
                    help='path to high resolution image dir')
parser.add_argument('--lr_img_dir', type=str, default=r'D:\workspace\dataset\OABreast\clipping\Neg_35_Left\LR',
                    help='path to desired output dir for downsampled images')
parser.add_argument('--sr_img_dir', type=str, default=r'D:\workspace\dataset\OABreast\clipping\Neg_35_Left\SR',
                    help='path to desired output dir for upsampled images')
parser.add_argument('--data_dir', type=str, default=r'',
                    help='总文件夹')
parser.add_argument('--is_2d', type=bool, default=True,
                    help='')
parser.add_argument('--is_OA-breast', type=bool, default=False,
                    help='')
parser.add_argument('--noise', type=bool, default=False,
                    help='')
parser.add_argument('--noise_level', type=float, default=0.1,
                    help='')
parser.add_argument('--num_of_DS', type=int, default=0,
                    help='多次bicubic上下采样')
parser.add_argument('--dist_of_DS', type=int, default=0,
                    help='没有使用')
parser.add_argument('--num_of_other', type=int, default=0,
                    help='多次隔点下取样，和bicubic上采样')
parser.add_argument('--mat_field', type=str, default=r'',
                    help='')
parser.add_argument('--data_range', type=int, default=0,
                    help='')
parser.add_argument('--nx', type=int)
parser.add_argument('--ny', type=int)
parser.add_argument('--nz', type=int)
args = parser.parse_args()

# 添加噪声
def add_noise(data):
    noise_level = args.noise_level
    noise = np.random.randn(*data.shape) * noise_level
    noisy_img = data + noise
    return noisy_img

# 规格化：收束像素范围
def quantize(img, data_range):
    img = img.astype(np.float64)
    img = img * 255 / data_range
    img = np.clip(img, 0, 255)
    img = img / 255 * data_range
    img = np.round(img).astype(np.uint8)
    return img

# bd下采样
def bd_img():
    hr_image_dir = args.hr_img_dir
    lr_image_dir = args.lr_img_dir

    print(args.hr_img_dir)
    print(args.lr_img_dir)

    # create LR image dirs
    os.makedirs(lr_image_dir + "/X2", exist_ok=True)
    os.makedirs(lr_image_dir + "/X3", exist_ok=True)
    os.makedirs(lr_image_dir + "/X4", exist_ok=True)
    os.makedirs(lr_image_dir + "/X6", exist_ok=True)

    supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                             ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                             ".tiff")

    # Downsample HR images
    for filename in os.listdir(hr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue

        name, ext = os.path.splitext(filename)

        # Read HR image
        hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
        # 我的疑惑：是不是写反了
        hr_img_dims = (hr_img.shape[1], hr_img.shape[0])

        # Blur with Gaussian kernel of width sigma = 1
        hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)
        # cv2.GaussianBlur(hr_img, (0,0), 1, 1)   其中模糊核这里用的0。两个1分别表示x、y方向的标准差。 可以具体查看该函数的官方文档。
        # Downsample image 2x
        lr_image_2x = cv2.resize(hr_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        if args.keepdims:
            lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + 'x2' + ext), lr_image_2x)

        # Downsample image 3x
        lr_img_3x = cv2.resize(hr_img, (0, 0), fx=(1 / 3), fy=(1 / 3),
                               interpolation=cv2.INTER_CUBIC)
        if args.keepdims:
            lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
                                   interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_image_dir + "/X3", filename.split('.')[0] + 'x3' + ext), lr_img_3x)

        # Downsample image 4x
        lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
                               interpolation=cv2.INTER_CUBIC)
        if args.keepdims:
            lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                                   interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_image_dir + "/X4", filename.split('.')[0] + 'x4' + ext), lr_img_4x)

        # Downsample image 6x
        lr_img_6x = cv2.resize(hr_img, (0, 0), fx=1 / 6, fy=1 / 6,
                               interpolation=cv2.INTER_CUBIC)
        if args.keepdims:
            lr_img_6x = cv2.resize(lr_img_6x, hr_img_dims,
                                   interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_image_dir + "/X6", filename.split('.')[0] + 'x6' + ext), lr_img_6x)

def bd_dat():
    hr_image_dir = args.hr_img_dir
    lr_image_dir = args.lr_img_dir
    nx = args.nx
    ny = args.ny
    nz = args.nz


    print(args.hr_img_dir)
    print(args.lr_img_dir)

    # create LR image dirs
    os.makedirs(lr_image_dir + "/X2", exist_ok=True)

    supported_img_formats = (".DAT")

    # Downsample HR images
    for filename in os.listdir(hr_image_dir):

        print(filename)

        if not filename.endswith(supported_img_formats):
            continue

        name, ext = os.path.splitext(filename)

        print(os.path.join(hr_image_dir, filename))

        # Read HR image
        hr_img = np.fromfile(os.path.join(hr_image_dir, filename), dtype=np.uint8)
        print(f"before shape:{np.shape(hr_img)}")
        hr_img = hr_img.reshape(nx, ny, nz)



        # Blur with Gaussian kernel of width sigma = 1
        for z in range(nz):
            item = hr_img[:, :, z]
            item = cv2.GaussianBlur(item, (0, 0), 1, 1)
            hr_img[:, :, z] = item

        print(f"after shape:{np.shape(hr_img)}")

        # cv2.GaussianBlur(hr_img, (0,0), 1, 1)   其中模糊核这里用的0。两个1分别表示x、y方向的标准差。 可以具体查看该函数的官方文档。
        # Downsample image 2x
        lr_image_2x = np.zeros((int(nx / 2), int(ny / 2), nz), dtype=np.uint8)
        for z in range(nz):
            item = hr_img[:, :, z]
            item = cv2.resize(item, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            lr_image_2x[:, :, z] = item

        print(f"after resize shape:{np.shape(lr_image_2x)}")
        lr_image_2x.tofile(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + 'x2' + ext))

# png图像上下采样
def bi_img_downsampling_x2():
    print("\nbi_img_downsampling_x2")
    hr_image_dir = args.hr_img_dir
    lr_image_dir = args.lr_img_dir

    print(hr_image_dir)
    print(lr_image_dir)

    # create LR image dirs
    os.makedirs(lr_image_dir + "/X2", exist_ok=True)

    supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                             ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                             ".tiff")

    # Downsample HR images
    for filename in os.listdir(hr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue
        name, ext = os.path.splitext(filename)

        # Read HR image
        hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
        # Downsample image 2x
        lr_image_2x = cv2.resize(hr_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + ext), lr_image_2x)

def bi_img_upsampling_x2():
    print("\nbi_img_upsampling_x2")
    sr_image_dir = args.sr_img_dir
    lr_image_dir = args.lr_img_dir

    if not lr_image_dir.endswith("X2"):
        lr_image_dir = os.path.join(lr_image_dir, 'X2')

    print(sr_image_dir)
    print(lr_image_dir)

    # create LR image dirs
    os.makedirs(sr_image_dir + "/X2", exist_ok=True)

    supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                             ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                             ".tiff")

    # Upsample HR images
    for filename in os.listdir(lr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue
        name, ext = os.path.splitext(filename)

        # Read HR image
        lr_img = cv2.imread(os.path.join(lr_image_dir, filename))
        # Downsample image 2x
        sr_image_2x = cv2.resize(lr_img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(sr_image_dir + "/X2", filename.split('.')[0] + ext), sr_image_2x)

# dat文件上下采样
def bi_dat_downsampling_x2():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_dat_downsampling_x2")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    hr_dir = os.path.join(args.data_dir, 'HR')
    hr_names = os.listdir(hr_dir)
    if len(hr_names) == 0:
        print(f"no hr files, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    print(f"hr_dir : {hr_dir}")
    print(f"lr_dir : {lr_dir}")
    # 获取三维
    dataset = re.split(r'[\\/]', args.data_dir)[-1]
    print(f"dataset : {dataset}")
    nx, ny, nz = get_3d(dataset)
    # 建立LR文件夹
    os.makedirs(lr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".DAT")

    # 下采样
    for filename in hr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取hr
        hr_img = np.fromfile(os.path.join(hr_dir, filename), dtype=np.uint8)
        hr_img = hr_img.reshape(nx, ny, nz)
        print(f"downsample before : {hr_img.shape}")
        # 进行下采样
        lr_img_2x = np.zeros((nx//2, ny//2, nz))
        for idx in range(nz):
            lr_img_2x[:, :, idx] = cv2.resize(hr_img[:, :, idx], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        print(f"downsample after : {lr_img_2x.shape}")
        """
        经过cv2.resize处理，dtype为浮点数，必须转换dtype
        """
        lr_img_2x = np.clip(0, 4, lr_img_2x)
        lr_img_2x = lr_img_2x.astype(np.uint8)
        # 保存lr文件
        lr_img_2x.tofile(os.path.join(lr_dir, filename))

def bi_dat_upsampling_x2():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_dat_downsampling_x2")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    lr_names = os.listdir(lr_dir)
    if len(lr_names) == 0:
        print(f"no lr files, function terminates")
        return
    sr_dir = os.path.join(args.data_dir, 'SR', 'X2')
    print(f"lr_dir : {lr_dir}")
    print(f"sr_dir : {sr_dir}")
    # 获取三维
    dataset = re.split(r'[\\/]', args.data_dir)[-1]
    print(f"dataset : {dataset}")
    nx, ny, nz = get_3d(dataset)
    # 建立sr文件夹
    os.makedirs(sr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".DAT")

    # 下采样
    for filename in lr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取lr
        lr_img = np.fromfile(os.path.join(lr_dir, filename), dtype=np.uint8)
        lr_img = lr_img.reshape(nx//2, ny//2, nz)
        print(f"downsample before : {lr_img.shape}")
        # 进行下采样
        sr_img_2x = np.zeros((nx, ny, nz))
        for idx in range(nz):
            sr_img_2x[:, :, idx] = cv2.resize(lr_img[:, :, idx], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        print(f"downsample after : {sr_img_2x.shape}")
        """
        经过cv2.resize处理，dtype为浮点数，必须转换dtype
        """
        sr_img_2x = np.clip(0, 4, sr_img_2x)
        sr_img_2x = sr_img_2x.astype(np.uint8)

        # 保存sr文件
        sr_img_2x.tofile(os.path.join(sr_dir, filename))

def bi_dat_downsampling_x2_3d():
    print("\nbi_dat_downsampling_x2_3d")
    hr_image_dir = os.path.join(args.data_dir, 'HR')
    lr_image_dir = os.path.join(args.data_dir, 'LR', "X2")
    print(f"sr_image_dir : {hr_image_dir}")
    print(f"lr_image_dir : {lr_image_dir}")
    # create LR image dirs
    os.makedirs(lr_image_dir, exist_ok=True)
    supported_img_formats = (".DAT")
    # 遍历HR文件夹下的文件
    for filename in os.listdir(hr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue
        # 确定三维
        nx, ny, nz = get_3d(os.path.splitext(filename)[0])
        # 获取图片
        hr_img = np.fromfile(os.path.join(hr_image_dir, filename), dtype=np.uint8)
        hr_img = hr_img.reshape(nx, ny, nz)
        # before shape
        print(hr_img.shape)
        # downsample
        """
        order
        0 : 最邻近插值
        1 : 双线性插值
        3 : 双三次插值
        """
        lr_img = zoom(hr_img, (0.5, 0.5, 0.5), order=3)
        lr_img =quantize(lr_img, 4)
        # after shape
        print(lr_img.shape)
        # 保存
        lr_img.tofile(os.path.join(lr_image_dir, filename))

def bi_dat_upsampling_x2_3d():
    print("\nbi_dat_upsampling_x2_3d")
    sr_image_dir = os.path.join(args.data_dir, 'SR', 'X2')
    lr_image_dir = os.path.join(args.data_dir, 'LR', 'X2')
    print(f"data_dir : {args.data_dir}")
    print(f"sr_image_dir : {sr_image_dir}")
    print(f"lr_image_dir : {lr_image_dir}")
    # create SR image dirs
    os.makedirs(sr_image_dir, exist_ok=True)
    supported_img_formats = (".DAT")
    # 遍历HR文件夹下的文件
    for filename in os.listdir(lr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue
        # 确定三维
        nx, ny, nz = get_3d(os.path.splitext(filename)[0])
        # 获取图片
        lr_img = np.fromfile(os.path.join(lr_image_dir, filename), dtype=np.uint8)
        lr_img = lr_img.reshape(nx//2, ny//2, nz)
        # before shape
        print(lr_img.shape)
        # upsample
        sr_img = zoom(lr_img, (2, 2, 2), order=3)
        sr_img = quantize(sr_img, 4)
        # after shape
        print(sr_img.shape)
        # save
        sr_img.tofile(os.path.join(sr_image_dir, filename))

# mat文件上下采样
def bi_mat_downsampling_x2():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_mat_downsampling_x2")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    hr_dir = os.path.join(args.data_dir, 'HR')
    hr_names = os.listdir(hr_dir)
    if len(hr_names) == 0:
        print(f"no hr files, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    print(f"hr_dir : {hr_dir}")
    print(f"lr_dir : {lr_dir}")
    # 建立LR文件夹
    os.makedirs(lr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".mat")
    # 下采样
    for filename in hr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取hr
        file = io.loadmat(os.path.join(hr_dir, filename))
        hr_img = file[args.mat_field]
        # 转换数据类型
        flag = (hr_img.dtype == np.uint16)
        flag1 = (hr_img.dtype == np.uint8)
        if flag or flag1:
            hr_img = hr_img.astype(np.float64)
        # 进行下采样
        shape = hr_img.shape
        print(f"downsample before : {shape}")
        # hist, bins = np.histogram(hr_img.flatten(),density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")
        lr_img_2x = np.zeros((shape[0]//2, shape[1]//2, shape[2]))
        for idx in range(shape[2]):
            lr_img_2x[:, :, idx] = cv2.resize(hr_img[:, :, idx], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        print(f"downsample after : {lr_img_2x.shape}")
        # hist, bins = np.histogram(lr_img_2x.flatten(),density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")
        """
        经过cv2.resize处理，dtype为浮点数
        """
        lr_img_2x = np.clip(0.0, args.data_range, lr_img_2x)
        # 转换数据类型
        if flag:
            lr_img_2x = lr_img_2x.astype(np.uint16)
        if flag1:
            lr_img_2x = lr_img_2x.astype(np.uint8)
        # 保存lr文件
        if args.mat_field == 'f1':
            io.savemat(os.path.join(lr_dir, filename), {'imgout' : lr_img_2x})
        elif args.mat_field == 'img':
            io.savemat(os.path.join(lr_dir, filename), {'img' : lr_img_2x})

def bi_mat_upsampling_x2():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_mat_upsampling_x2")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    lr_names = os.listdir(lr_dir)
    if len(lr_names) == 0:
        print(f"no lr files, function terminates")
        return
    sr_dir = os.path.join(args.data_dir, 'SR', 'X2')
    print(f"lr_dir : {lr_dir}")
    print(f"sr_dir : {sr_dir}")
    # 建立sr文件夹
    os.makedirs(sr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".mat")
    # 下采样
    for filename in lr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取lr
        file = io.loadmat(os.path.join(lr_dir, filename))
        lr_img = file[args.mat_field]
        # 转换数据类型
        flag = (lr_img.dtype == np.uint16)
        flag1 = (lr_img.dtype == np.uint8)
        if flag or flag1:
            lr_img = lr_img.astype(np.float64)
        # 进行下采样
        shape = lr_img.shape
        print(f"downsample before : {shape}")

        # hist, bins = np.histogram(lr_img.flatten(), density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")

        sr_img_2x = np.zeros((shape[0]*2, shape[1]*2, shape[2]))
        for idx in range(shape[2]):
            sr_img_2x[:, :, idx] = cv2.resize(lr_img[:, :, idx], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        print(f"downsample after : {sr_img_2x.shape}")

        # hist, bins = np.histogram(sr_img_2x.flatten(), density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")

        """
        经过cv2.resize处理，dtype为浮点数
        """
        sr_img_2x = np.clip(0.0, args.data_range, sr_img_2x)
        # 转换数据类型
        if flag:
            sr_img_2x = sr_img_2x.astype(np.uint16)
        if flag1:
            sr_img_2x = sr_img_2x.astype(np.uint8)
        # 保存sr文件
        if args.mat_field == 'imgout':
            io.savemat(os.path.join(sr_dir, filename), {'f1' : sr_img_2x})
        elif args.mat_field == 'img':
            io.savemat(os.path.join(sr_dir, filename), {'img' : sr_img_2x})

# 设置多次下采样，和每次下采样距离，达到控制psnr的效果
def bi_mat_downsampling_x2_3d():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_mat_downsampling_x2_3d")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    hr_dir = os.path.join(args.data_dir, 'HR')
    hr_names = os.listdir(hr_dir)
    if len(hr_names) == 0:
        print(f"no hr files, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    print(f"hr_dir : {hr_dir}")
    print(f"lr_dir : {lr_dir}")
    # 建立LR文件夹
    os.makedirs(lr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".mat")
    # 下采样
    for filename in hr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取hr
        file = io.loadmat(os.path.join(hr_dir, filename))
        hr_img = file['f1']
        # 转换数据类型
        flag = (hr_img.dtype == np.uint16)
        if flag:
            hr_img.astype(np.float64)
        # 进行下采样
        shape = hr_img.shape
        print(f"downsample before : {shape}")

        # hist, bins = np.histogram(hr_img.flatten(),density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")

        # lr_img_2x = np.zeros((shape[0]//2, shape[1]//2, shape[2]))
        # for idx in range(shape[2]):
        #     lr_img_2x[:, :, idx] = cv2.resize(hr_img[:, :, idx], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        if args.num_of_DS == 0:
            lr_img_2x = zoom(hr_img, (0.5, 0.5, 0.5), order=3)
        else:
            temp = hr_img
            for idx in range(args.num_of_DS-1):
                print(f'repeat down and up sampling: {idx}')
                temp = zoom(temp, (0.5, 0.5, 0.5), order=3)
                temp = zoom(temp, (2, 2, 2), order=3)
            lr_img_2x = zoom(temp, (0.5, 0.5, 0.5), order=3)

        print(f"downsample after : {lr_img_2x.shape}")

        # hist, bins = np.histogram(lr_img_2x.flatten(),density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")

        """
        经过cv2.resize处理，dtype为浮点数
        """
        lr_img_2x = np.clip(0.0, 1.0e3, lr_img_2x)
        # 转换数据类型
        if flag:
            lr_img_2x.astype(np.float64)
        # 保存lr文件
        io.savemat(os.path.join(lr_dir, filename), {'imgout' : lr_img_2x})

def bi_mat_upsampling_x2_3d():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_mat_upsampling_x2_3d")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    lr_names = os.listdir(lr_dir)
    if len(lr_names) == 0:
        print(f"no lr files, function terminates")
        return
    sr_dir = os.path.join(args.data_dir, 'SR', 'X2')
    print(f"lr_dir : {lr_dir}")
    print(f"sr_dir : {sr_dir}")
    # 建立sr文件夹
    os.makedirs(sr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".mat")
    # 下采样
    for filename in lr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取lr
        file = io.loadmat(os.path.join(lr_dir, filename))
        lr_img = file['imgout']
        # 转换数据类型
        flag = (lr_img.dtype == np.uint16)
        if flag:
            lr_img.astype(np.float64)
        # 进行下采样
        shape = lr_img.shape
        print(f"downsample before : {shape}")

        # hist, bins = np.histogram(lr_img.flatten(),density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")

        # sr_img_2x = np.zeros((shape[0]//2, shape[1]//2, shape[2]))
        # for idx in range(shape[2]):
        #     sr_img_2x[:, :, idx] = cv2.resize(lr_img[:, :, idx], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        sr_img_2x = zoom(lr_img, (2, 2, 2), order=3)
        print(f"downsample after : {sr_img_2x.shape}")

        # hist, bins = np.histogram(sr_img_2x.flatten(),density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")

        """
        经过cv2.resize处理，dtype为浮点数
        """
        sr_img_2x = np.clip(0, 1.0e3, sr_img_2x)
        # 转换数据类型
        if flag:
            sr_img_2x.astype(np.float64)
        # 保存sr文件
        io.savemat(os.path.join(sr_dir, filename), {'f1' : sr_img_2x})

# mat文件隔点下采样
def mat_downsampling_x2_every_other_point():
    """
    间隔取点
    提供data_dir
    """
    print("\nbi_mat_downsampling_x2_every_other_point")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    hr_dir = os.path.join(args.data_dir, 'HR')
    hr_names = os.listdir(hr_dir)
    if len(hr_names) == 0:
        print(f"no hr files, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    print(f"hr_dir : {hr_dir}")
    print(f"lr_dir : {lr_dir}")
    # 建立LR文件夹
    os.makedirs(lr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".mat")
    # 开始下采样
    for filename in hr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取hr
        file = io.loadmat(os.path.join(hr_dir, filename))
        hr_img = file['f1']
        # 转换数据类型
        flag = (hr_img.dtype == np.uint16)
        if flag:
            hr_img.astype(np.float64)
        # 构建lr容器
        shape = hr_img.shape
        print(f"downsample before : {shape}")
        # hist, bins = np.histogram(hr_img.flatten(), density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")
        lr_img_2x = np.zeros((shape[0] // 2, shape[1] // 2, shape[2]))
        # downsampling
        for ih in range(0, shape[0], 2):
            for iw in range(0, shape[1], 2):
                for idepth in range(shape[2]):
                    lr_img_2x[ih//2, iw//2, idepth] = hr_img[ih, iw, idepth]
        # for idx in range(shape[2]):
        #     lr_img_2x[:, :, idx] = cv2.resize(hr_img[:, :, idx], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        print(f"downsample after : {lr_img_2x.shape}")
        # hist, bins = np.histogram(lr_img_2x.flatten(), density=True)
        # cumhist = np.cumsum(hist)
        # print(f"hist : {hist}")
        # print(f"cumhist : {cumhist}")
        # print(f"bins : {bins}")
        """
        经过cv2.resize处理，dtype为浮点数
        """
        lr_img_2x = np.clip(0.0, 1.0e3, lr_img_2x)
        # 转换数据类型
        if flag:
            lr_img_2x.astype(np.float64)
        # 保存lr文件
        io.savemat(os.path.join(lr_dir, filename), {'imgout': lr_img_2x})
# 修改：添加噪声
# 设置多次隔点取样，搭配bicubic上采样
def mat_downsampling_x2_3d_every_other_point():
    """
    间隔取点
    提供data_dir
    """
    print("\nbi_mat_downsampling_x2_3d_every_other_point")
    # 判定path是否合理
    print(f"data_dir : {args.data_dir}")
    if not os.path.isdir(args.data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    hr_dir = os.path.join(args.data_dir, 'HR')
    hr_names = os.listdir(hr_dir)
    if len(hr_names) == 0:
        print(f"no hr files, function terminates")
        return
    lr_dir = os.path.join(args.data_dir, 'LR', 'X2')
    print(f"hr_dir : {hr_dir}")
    print(f"lr_dir : {lr_dir}")
    # 建立LR文件夹
    os.makedirs(lr_dir, exist_ok=True)
    # 支持文件类型
    supported_img_formats = (".mat")
    # 开始下采样
    for filename in hr_names:
        print(filename)
        if not filename.endswith(supported_img_formats):
            continue
        # 获取hr
        file = io.loadmat(os.path.join(hr_dir, filename))
        hr_img = file['f1']
        # 转换数据类型
        flag = (hr_img.dtype == np.uint16)
        if flag:
            hr_img.astype(np.float64)
        # 构建lr容器
        shape = hr_img.shape
        print(f"downsample before : {shape}")
        lr_img_2x = np.zeros((shape[0] // 2, shape[1] // 2, shape[2]//2))
        # downsampling
        if args.num_of_other == 0 or args.num_of_other == 1:
            for ih in range(0, shape[0], 2):
                for iw in range(0, shape[1], 2):
                    for idepth in range(0, shape[2], 2):
                        lr_img_2x[ih//2, iw//2, idepth//2] = hr_img[ih, iw, idepth]
        else:
            temp_hr = hr_img
            temp_lr = np.zeros((shape[0] // 2, shape[1] // 2, shape[2] // 2))
            for idx in range(args.num_of_other-1):
                print(f'repeat down and up sampling: {idx}')
                # 隔点下采样temp_hr
                for ih in range(0, shape[0], 2):
                    for iw in range(0, shape[1], 2):
                        for idepth in range(0, shape[2], 2):
                            temp_lr[ih // 2, iw // 2, idepth // 2] = temp_hr[ih, iw, idepth]
                # bicubic上采样temp_lr
                temp_hr = zoom(temp_lr, (2, 2, 2), order=3)
            # 隔点下采样temp_hr，获取最终的lr_img_2x
            for ih in range(0, shape[0], 2):
                for iw in range(0, shape[1], 2):
                    for idepth in range(0, shape[2], 2):
                        lr_img_2x[ih // 2, iw // 2, idepth // 2] = temp_hr[ih, iw, idepth]
        # 添加噪声
        if args.noise:
            lr_img_2x = add_noise(lr_img_2x)

        print(f"downsample after : {lr_img_2x.shape}")
        """
        经过cv2.resize处理，dtype为浮点数
        """
        lr_img_2x = np.clip(0.0, 1.0e3, lr_img_2x)
        # 转换数据类型
        if flag:
            lr_img_2x.astype(np.float64)
        # 保存lr文件
        io.savemat(os.path.join(lr_dir, filename), {'imgout': lr_img_2x})

# mat文件（USCT数据转OA-breast数据）上下采样
def bi_mat_to_oabreast_downsampling_x2_3d():
    print("\nbi_mat_to_oabreast_downsampling_x2_3d")
    hr_image_dir = os.path.join(args.data_dir, 'HR')
    lr_image_dir = os.path.join(args.data_dir, 'LR', "X2")
    print(f"sr_image_dir : {hr_image_dir}")
    print(f"lr_image_dir : {lr_image_dir}")
    # create LR image dirs
    os.makedirs(lr_image_dir, exist_ok=True)
    # supported_img_formats = (".DAT")
    supported_img_formats = (".mat")
    # 遍历HR文件夹下的文件
    for filename in os.listdir(hr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue
        # 获取图片
        # file = np.fromfile(os.path.join(hr_image_dir, filename), dtype=np.uint8)
        file = io.loadmat(os.path.join(hr_image_dir, filename))
        hr_img = file['f1']
        # before shape
        print(hr_img.shape)
        # downsample
        """
        order
        0 : 最邻近插值
        1 : 双线性插值
        3 : 双三次插值
        """
        lr_img = zoom(hr_img, (0.5, 0.5, 0.5), order=3)
        lr_img =quantize(lr_img, 4)
        # after shape
        print(lr_img.shape)
        # 保存
        # lr_img.tofile(os.path.join(lr_image_dir, filename))
        io.savemat(os.path.join(lr_image_dir, filename), {'imgout': lr_img})

def bi_mat_to_oabreast_upsampling_x2_3d():
    print("\nbi_dat_upsampling_x2_3d")
    sr_image_dir = os.path.join(args.data_dir, 'SR', 'X2')
    lr_image_dir = os.path.join(args.data_dir, 'LR', 'X2')
    # print(f"data_dir : {args.data_dir}")
    print(f"sr_image_dir : {sr_image_dir}")
    print(f"lr_image_dir : {lr_image_dir}")
    # create SR image dirs
    os.makedirs(sr_image_dir, exist_ok=True)
    # supported_img_formats = (".DAT")
    supported_img_formats = (".mat")
    # 遍历HR文件夹下的文件
    for filename in os.listdir(lr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue
        # 获取图片
        # file = np.fromfile(os.path.join(lr_image_dir, filename), dtype=np.uint8)
        file = io.loadmat(os.path.join(lr_image_dir, filename))
        lr_img = file['imgout']
        # before shape
        print(lr_img.shape)
        # upsample
        sr_img = zoom(lr_img, (2, 2, 2), order=3)
        sr_img = quantize(sr_img, 4)
        # after shape
        print(sr_img.shape)
        # save
        # sr_img.tofile(os.path.join(sr_image_dir, filename))
        io.savemat(os.path.join(sr_image_dir, filename), {'f1': sr_img})

if __name__ == '__main__':
    # png图片
    # args.hr_img_dir = 'D:\workspace\dataset\Manga109\clipping\HR'
    # args.lr_img_dir = r'D:\workspace\dataset\Urban100\LR'
    # args.sr_img_dir = r'D:\workspace\dataset\Urban100\SR'
    # bi_img_downsampling_x2()
    # bi_img_upsampling_x2()

    # datasets = ['Set5', 'Set14']
    # prefix = r'D:\workspace\dataset'
    # suffix_lr = r'LR'
    # suffix_sr = r'SR'
    # for dataset in datasets:
    #     args.dataset = dataset
    #     args.lr_img_dir = os.path.join(prefix, dataset, suffix_lr)
    #     args.sr_img_dir = os.path.join(prefix, dataset, suffix_sr)
    #     bi_img_upsampling_x2()

    # oabreast 2d
    # 提供文件夹
    # d1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    # d2 = r"_Left"
    # datasets = ['07', '35', '47']
    # suffixes = ['', '_train', '_test']
    # for idx_dataset in range(3):
    #     for idx_suffix in range(3):
    #         args.data_dir = d1 + datasets[idx_dataset] + d2 + suffixes[idx_suffix]
    #         bi_dat_downsampling_x2()
    #         bi_dat_upsampling_x2()

    path = r'D:\workspace\dataset\OABreast\dat2mat\clipping\pixel_translation'
    for foldername in os.listdir(path):
        args.data_dir = os.path.join(path, foldername)
        args.mat_field = 'img'
        args.data_range = 4
        bi_mat_downsampling_x2()
        bi_mat_upsampling_x2()

    # # oabreast 3d
    # # 只需要提供文件夹路径
    # args.data_dir = r'D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\temp'
    # bi_dat_downsampling_x2_3d()
    # bi_dat_upsampling_x2_3d()

    # # usct 2d
    # # 提供文件夹路径
    # path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\every_other_points_2d_float'
    # for foldername in os.listdir(path):
    #     if foldername != 'HR':
    #         args.data_dir = os.path.join(path, foldername)
    #         mat_downsampling_x2_every_other_point()
    #         bi_mat_upsampling_x2()
    #
    # # usct 3d
    # # 提供文件夹路径
    # args.data_dir = r'/root/autodl-tmp/dataset/USCT_3d/every_other_points_3d_float_test/'
    # # bi_mat_downsampling_x2_3d()
    # mat_downsampling_x2_3d_every_other_point()
    # bi_mat_upsampling_x2_3d()

    # usct to oabreast 3d
    # 提供文件夹路径

    # nums = [5]
    # for args.num_of_other in nums:
    #     args.data_dir = rf'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_low_{args.num_of_other}'
    #     # args.data_dir = rf'D:\workspace\dataset\USCT\clipping\pixel_translation\every_other_points_3d_float_{args.num_of_other}'
    #     mat_downsampling_x2_3d_every_other_point()
    #     bi_mat_upsampling_x2_3d()

    # args.num_of_other = 8
    # args.data_dir = rf'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_low_{args.num_of_other}'
    # # args.data_dir = rf'D:\workspace\dataset\USCT\clipping\pixel_translation\every_other_points_3d_float_{args.num_of_other}'
    # mat_downsampling_x2_3d_every_other_point()
    # bi_mat_upsampling_x2_3d()

    # args.data_dir = r'/root/autodl-tmp/dataset/USCT_3d/3d_to_oabreast'
    # # bi_mat_to_oabreast_downsampling_x2_3d()
    # bi_mat_to_oabreast_upsampling_x2_3d()


