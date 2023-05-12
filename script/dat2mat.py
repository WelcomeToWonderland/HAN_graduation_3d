import os
import cv2
import numpy as np
import argparse
from src.utility import get_3d_unmodified
from scipy import io

parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--path_dat', type=str, default=r'',
                    help='')
parser.add_argument('--path_mat', type=str, default=r'',
                    help='')
parser.add_argument('--basename', type=str, default=r'',
                    help='')
args = parser.parse_args()

def dat2mat():
    """
    文件夹路径
    :return:
    """
    # 创建保存文件夹
    print(f"\npath dat : {args.path_dat}")
    print(f"path mat : {args.path_mat}")
    os.makedirs(args.path_mat, exist_ok=True)
    # 遍历文件夹下文件
    for filename in os.listdir(args.path_dat):
        # 拼接路径
        path_dat = os.path.join(args.path_dat, filename)
        # 加载数据
        data = np.fromfile(path_dat, dtype=np.uint8)
        x, y, z = get_3d_unmodified(args.basename)
        data = data.reshape(x, y, z)
        # 转换文件类型
        file_dat = {'img': data}
        path_mat = os.path.join(args.path_mat, basename + '.mat')
        io.savemat(path_mat, file_dat)


if __name__ == '__main__':
    prefix_dat = r'D:\workspace\dataset\OABreast\original'
    prefix_mat = r'D:\workspace\dataset\OABreast\dat2mat'
    suffix_mat = r'HR'
    basenames = ['Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left', 'Neg_07_Left_train', 'Neg_07_Left_test']
    for idx in range(3):
        basename = basenames[idx]
        args.basename = basename
        args.path_dat = os.path.join(prefix_dat, basename)
        args.path_mat = os.path.join(prefix_mat, basename, suffix_mat)
        dat2mat()

