'''
通过剪裁图像，将图像前两个维度中奇数转变为偶数
防止图像下采样之后，上采样的图形与原图像shape不一致
'''
import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--path', type=str, default=r'',
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
    # 只对HR处理
    # 获取文件
    args.path = os.path.join(args.path, 'HR','MergedPhantom.DAT')
    data = np.fromfile(args.path, dtype=np.uint8)
    data = data.reshape(args.nx, args.ny, args.nz)
    print(f'\nbefore clipping : {data.shape}')
    for idx in range(3):
        if(data.shape[idx] % 2 != 0):
            data = np.delete(data, 0, axis=idx)
    print(f'\nafter clipping : {data.shape}')
    data.tofile(args.path)


if __name__ == '__main__':
    path = r'D:\workspace\dataset\OABreast\clipping\pixel_translation\downing'
    nxs = [616, 284, 494]
    nys = [484, 410, 614]
    """
    original
    train
    test
    """
    nzs = [719, 722, 752,
           319, 322, 352,
           400, 400, 400]

    for filename in os.listdir(path):
        if filename.split('_')[0] != 'Neg':
            continue
        args.path = os.path.join(path, filename)
        if args.path.split('_')[-2] == '07':
            idx = 0
        elif args.path.split('_')[-2] == '35':
            idx = 1
        elif args.path.split('_')[-2] == '47':
            idx = 2
        if args.path.endswith('train'):
            multiple = 1
        elif args.path.endswith('test'):
            multiple = 2
        else:
            multiple = 0
        args.nx = nxs[idx]
        args.ny = nys[idx]
        args.nz = nzs[3 * multiple + idx]
        clipping_dat()


