import os
import argparse
import cv2
import numpy as np


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
parser.add_argument('--nx', type=int)
parser.add_argument('--ny', type=int)
parser.add_argument('--nz', type=int)
args = parser.parse_args()

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

def bi_dat_downsampling_x2():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_dat_downsampling_x2")
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
        # Read HR image
        hr_img = np.fromfile(os.path.join(hr_image_dir, filename), dtype=np.uint8)
        hr_img = hr_img.reshape(nx, ny, nz)
        print(f"downsample before : {hr_img.shape}")
        print(f"hr_size : {hr_img.size}")
        hist, bins = np.histogram(hr_img, bins=range(7))
        densities, _ = np.histogram(hr_img, bins=range(7), density=True)
        densities_cumulative = densities.cumsum()
        print(f"hist : {hist}")
        print(f"densities : {densities}")
        print(f"densities_cumulative : {densities_cumulative}")
        print(f"bins : {bins}")
        # hr_img = hr_img.astype(np.float32)
        # hr_img = hr_img.astype(np.float32) * 255 / 4
        # Downsample image 2x
        lr_image_2x = np.zeros((int(nx / 2), int(ny / 2), nz))
        for idx in range(nz):
            lr_image_2x[:, :, idx] = cv2.resize(hr_img[:, :, idx], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            if(idx == nz -1):
                print(f"downsampling")
                print(f"idx = nz -1, size : {lr_image_2x[:, :, idx].size}")
                print(f"idx = nz -1, num_0 : {lr_image_2x[:, :, idx].size - np.count_nonzero(lr_image_2x[:, :, idx])}")
                print(f"idx = nz -1, max : {np.amax(lr_image_2x[:, :, idx])}")
        print(f"downsample after : {lr_image_2x.shape}")
        print(f"before normalize")
        print(f"lr_size : {lr_image_2x.size}")
        print(f"lr_num_0 : {lr_image_2x.size - np.count_nonzero(lr_image_2x)}")
        print(f"lr_max : {np.amax(lr_image_2x)}")

        # lr_image_2x = cv2.normalize(lr_image_2x, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 4

        # lr_image_2x = lr_image_2x * 255 / 4
        # lr_image_2x = np.clip(lr_image_2x, 0, 255)
        # lr_image_2x = lr_image_2x / 255 * 4

        # lr_image_2x = np.clip(lr_image_2x, 0, 255)
        # lr_image_2x = lr_image_2x / 255 * 4

        # lr_image_2x = np.round(lr_image_2x)

        """
        经过cv2.resize处理，dtypee为浮点数，必须转换dtype
        """
        lr_image_2x = lr_image_2x.astype(np.uint8)

        print(f"after normalize")
        print(f"lr_size : {lr_image_2x.size}")
        hist, bins = np.histogram(lr_image_2x, range(7))
        densities, _ = np.histogram(lr_image_2x, range(7), density=True)
        densities_cumulative = densities.cumsum()
        print(f"hist : {hist}")
        print(f"densities : {densities}")
        print(f"densities_cumulative : {densities_cumulative}")
        print(f"bins : {bins}")

        print(lr_image_2x.dtype)

        lr_image_2x.tofile(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + ext))

def bi_dat_upsampling_x2():
    """
    BI : 仅bicubic
    :return:
    """
    print("\nbi_dat_upsampling_x2")
    sr_image_dir = args.sr_img_dir
    lr_image_dir = args.lr_img_dir
    if not lr_image_dir.endswith("X2"):
        lr_image_dir = os.path.join(lr_image_dir, 'X2')
    nx = int(args.nx / 2)
    ny = int(args.ny / 2)
    nz = args.nz
    # nx = args.nx
    # ny = args.ny
    # nz = args.nz
    print(sr_image_dir)
    print(lr_image_dir)

    # create LR image dirs
    os.makedirs(sr_image_dir + "/X2", exist_ok=True)

    supported_img_formats = (".DAT")

    # Downsample HR images
    for filename in os.listdir(lr_image_dir):
        print(filename)

        if not filename.endswith(supported_img_formats):
            continue

        name, ext = os.path.splitext(filename)
        
        # Read HR image
        lr_img = np.fromfile(os.path.join(lr_image_dir, filename), dtype=np.uint8)
        lr_img = lr_img.reshape(nx, ny, nz)
        print(f"upsampling before : {lr_img.shape}")
        print(f"lr_size : {lr_img.size}")
        hist, bins = np.histogram(lr_img, bins=range(7))
        densities, _ = np.histogram(lr_img, bins=range(7), density=True)
        densities_cumulative = densities.cumsum()
        print(f"hist : {hist}")
        print(f"densities : {densities}")
        print(f"densities_cumulative : {densities_cumulative}")
        print(f"bins : {bins}")
        # lr_img = lr_img.astype(np.float32)
        # lr_img = lr_img.astype(np.float32) * 255 / 4
        # upsample image 2x
        sr_image_2x = np.zeros((nx*2, ny*2, nz))
        for idx in range(nz):
            sr_image_2x[:, :, idx] = cv2.resize(lr_img[:, :, idx], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # sr_image_2x = cv2.normalize(sr_image_2x, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 4

        # sr_image_2x = sr_image_2x * 255 / 4
        # sr_image_2x = np.clip(sr_image_2x, 0, 255)
        # sr_image_2x = sr_image_2x / 255 * 4

        # sr_image_2x = np.clip(sr_image_2x, 0, 255)
        # sr_image_2x = sr_image_2x / 255 * 4

        # sr_image_2x = np.round(sr_image_2x)

        sr_image_2x = sr_image_2x.astype(np.uint8)

        print(f"upsampling after : {sr_image_2x.shape}")
        print(f"sr_size : {sr_image_2x.size}")
        hist, bins = np.histogram(sr_image_2x, bins=range(7))
        densities, _ = np.histogram(sr_image_2x, bins=range(7), density=True)
        densities_cumulative = densities.cumsum()
        print(f"hist : {hist}")
        print(f"densities : {densities}")
        print(f"densities_cumulative : {densities_cumulative}")
        print(f"bins : {bins}")

        print(f"after resize shape:{np.shape(sr_image_2x)}")
        sr_image_2x.tofile(os.path.join(sr_image_dir + "/X2", filename.split('.')[0] + ext))

if __name__ == '__main__':
    d1 = 'OABreast_Neg_'
    d2 = '_Left'
    h1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    h2 = r"_Left\HR"
    l1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    l2 = r"_Left\LR"
    s1 = r"D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_"
    s2 = r"_Left\SR"
    datasets = ['07', '35', '47']
    nxs = [616, 284, 494]
    nys = [484, 410, 614]
    nzs = [719, 722, 752]
    for idx in range(3):
        args.dataset = d1 + datasets[idx] + d2
        args.hr_img_dir = h1 + datasets[idx] + h2
        args.lr_img_dir = l1 + datasets[idx] + l2
        args.sr_img_dir = s1 + datasets[idx] + s2
        args.nx = nxs[idx]
        args.ny = nys[idx]
        args.nz = nzs[idx]
        bi_dat_downsampling_x2()
        bi_dat_upsampling_x2()

    # args.hr_img_dir = 'D:\workspace\dataset\Manga109\clipping\HR'
    # args.lr_img_dir = 'D:\workspace\dataset\Manga109\clipping\LR'
    # args.sr_img_dir = 'D:\workspace\dataset\Manga109\clipping\SR'
    # bi_img_downsampling_x2()
    # bi_img_upsampling_x2()



