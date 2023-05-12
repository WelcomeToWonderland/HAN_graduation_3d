import numpy as np
import os
import argparse
from scipy import io

parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--translation', type=int, default=0,
                    help='')
args = parser.parse_args()

def downing(original, modified):
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    file = np.fromfile(original, dtype=np.uint8)
    print("before downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    for pixel in range(2, 6):
        file = np.where(file == pixel, pixel-1, file)
    print("after downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    file.tofile(modified)

def uping(original, modified):
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    file = np.fromfile(original, dtype=np.uint8)
    print("before downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    for pixel in range(1, 5):
        file = np.where(file == pixel, pixel+1, file)
    print("after downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    file.tofile(modified)

def downing_oabreast(original, modified):
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    # file = np.fromfile(original, dtype=np.uint8)
    file = io.loadmat(original)
    file = file['img']
    print("before downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    for pixel in range(2, 6):
        file = np.where(file == pixel, pixel-1, file)
    print("after downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    # file.tofile(modified)
    io.savemat(modified, {'img': file})

def uping_oabreast(original, modified):
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    file = np.fromfile(original, dtype=np.uint8)
    print("before downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    for pixel in range(1, 5):
        file = np.where(file == pixel, pixel+1, file)
    print("after downing")
    hist, bins = np.histogram(file, range(7))
    dentities, _ = np.histogram(file, range(7), density=True)
    print(f"hist : {hist}")
    print(f"dentities : {dentities}")
    print(f"bins : {bins}")
    file.tofile(modified)

def downing_mat(original, modified):
    """
    输入文件路径
    :param original:
    :param modified:
    :return:
    """
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    # 创建输出文件夹
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    # 读取文件
    file = io.loadmat(original)
    data = file['f1']
    print("before downing")
    print(f"min : {data.min()}")
    print(f"max : {data.max()}")
    data -= args.translation
    print("after downing")
    print(f"min : {data.min()}")
    print(f"max : {data.max()}")
    file['f1'] = data
    io.savemat(modified, file)

def uping_mat(original, modified):
    """
    输入src和dst文件路径
    分析：
    文件夹结构千奇百怪，直接限定文件路径
    :param original:
    :param modified:
    :return:
    """
    print(f"\noriginal : {original}")
    print(f"modified : {modified}")
    # 创建输出文件夹
    folder, _ = os.path.split(modified)
    print(f"modified folder : {folder}")
    os.makedirs(folder, exist_ok=True)
    # 读取文件
    file = io.loadmat(original)
    data = file['f1']
    print("before downing")
    print(f"min : {data.min()}")
    print(f"max : {data.max()}")
    data += args.translation
    print("after downing")
    print(f"min : {data.min()}")
    print(f"max : {data.max()}")
    file['f1'] = data
    io.savemat(modified, file)

if __name__ == '__main__':
    o1 = r"D:\workspace\dataset\OABreast\dat2mat\clipping"
    o2 = r"HR"
    m1 = r"D:\workspace\dataset\OABreast\dat2mat\clipping\pixel_translation"
    m2 = r"HR"
    basenames = ['Neg_07_Left', 'Neg_35_Left', 'Neg_47_Left', 'Neg_07_Left_train', 'Neg_07_Left_test']
    for idx in range(5):
        basename = basenames[idx]
        original = os.path.join(o1, basename, o2, basename + '.mat')
        modified = os.path.join(m1, basename, m2, basename + '.mat')
        downing_oabreast(original, modified)


    # args.translation = 1000
    # src = r'D:\workspace\dataset\USCT\clipping'
    # dst = r'D:\workspace\dataset\USCT\clipping\pixel_translation'
    # # 2d
    # temp_src = os.path.join(src, '2d')
    # temp_dst = os.path.join(dst, '2d')
    # for foldername in os.listdir(temp_src):
    #     # 确定文件路径
    #     original = os.path.join(temp_src, foldername, 'HR', foldername+'.mat')
    #     modified = os.path.join(temp_dst, foldername, 'HR', foldername+'.mat')
    #     downing_mat(original, modified)
    # # 3d
    # temp_src = os.path.join(src, '3d', 'HR')
    # temp_dst = os.path.join(dst, '3d', 'HR')
    # for filename in os.listdir(temp_src):
    #     original = os.path.join(temp_src, filename)
    #     modified = os.path.join(temp_dst, filename)
    #     downing_mat(original, modified)

