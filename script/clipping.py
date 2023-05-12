import os
import cv2
import numpy as np
import argparse
from src.utility import get_3d_unmodified
from scipy import io

parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--path_original', type=str, default=r'',
                    help='')
parser.add_argument('--path_clipping', type=str, default=r'',
                    help='')
parser.add_argument('--basename', type=str, default=r'',
                    help='')
parser.add_argument('--nx', type=int)
parser.add_argument('--ny', type=int)
parser.add_argument('--nz', type=int)
args = parser.parse_args()

def clipping_img(path):
    '''
    将一个数据集文件夹中的所有图片的奇数维度减一，转化为偶数
    最后处理过的文件替换原文件
    '''
    path_dataset = path
    name_img_list = os.listdir(path_dataset)

    for name_img in name_img_list:
        img = cv2.imread(os.path.join(path_dataset, name_img))
        shape_img = np.shape(img)

        print(f"name:{name_img}, shape:{shape_img}")

        clipping = 0
        if shape_img[0] % 2 == 1:
            clipping = 1
            img = np.delete(img, 0, 0)
        if shape_img[1] % 2 == 1:
            clipping = 1
            img = np.delete(img, 0, 1)

        if clipping == 1:
            print(f"shape_img:{shape_img}, shape_clipping:{np.shape(img)}")
            cv2.imwrite(os.path.join(path_dataset, name_img), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def clipping_dat():
    # 固定的文件名
    filename = 'MergedPhantom.DAT'
    # 拼接加载路径
    path_original = os.path.join(args.path_original, filename)
    # 读取数据，reshape数据
    data = np.fromfile(path_original, dtype=np.uint8)
    x, y, z = get_3d_unmodified(args.basename)
    data = data.reshape(x, y, z)
    # clipping
    print(f'\nbefore clipping : {data.shape}')
    for idx in range(3):
        if(data.shape[idx] % 2 != 0):
            # data = np.delete(data, 0, axis=idx)
            if idx == 0:
                data = data[1:]
            elif idx == 1:
                data = data[:, 1:]
            elif idx == 2:
                data = data[:, :, 1:]
    print(f'\nafter clipping : {data.shape}')
    # 存储clipping后的data
    os.makedirs(args.path_clipping, exist_ok=True)
    path_clipping = os.path.join(args.path_clipping, filename)
    data.tofile(path_clipping)

def clipping_mat():
    """
    输入文件夹路径，批量处理文件
    将三个维度，都处理成偶数
    :return:
    """
    original = args.path_original
    clipping = args.path_clipping
    # 检验original文件夹路径合法性
    if not os.path.isdir(original):
        print("original folder doesn`t exist")
        return
    # 创建clipping文件夹
    os.makedirs(clipping, exist_ok=True)
    print(f"\noriginal : {original}")
    print(f"clipping : {clipping}")
    # 遍历处理所有文件
    for filename in os.listdir(original):
        print(filename)
        file = io.loadmat(os.path.join(original, filename))
        data = file['f1']
        print(f"before clipping : {data.shape}")
        for idx in range(3):
            if data.shape[idx]%2!=0:
                data = np.delete(data, 0, axis=idx)
        print(f"after clipping : {data.shape}")
        # 存储文件
        file['f1'] = data
        io.savemat(os.path.join(clipping, filename), file)




if __name__ == '__main__':
    # args.path_original = r'D:\workspace\dataset\USCT\original\HR'
    # args.path_clipping = r'D:\workspace\dataset\USCT\clipping\HR'
    # clipping_mat()

    prefix_original = r'D:\workspace\dataset\OABreast\original'
    prefix_clipping = r'D:\workspace\dataset\OABreast\clipping'
    suffix_clipping = r'HR'
    basenames = ['Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left', 'Neg_07_Left_train', 'Neg_07_Left_test']
    for idx in range(3):
        basename = basenames[idx]
        args.basename = basename
        args.path_original = os.path.join(prefix_original, basename)
        args.path_clipping = os.path.join(prefix_clipping, basename, suffix_clipping)
        clipping_dat()

