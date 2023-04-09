import os
import argparse
import cv2
import numpy as np
from src.utility import get_3d

# parse args
parser = argparse.ArgumentParser(description='extract testset from dataset form 300 ~ 499')
parser.add_argument('--path', type=str, default=r'D:\workspace\dataset\OABreast\clipping\Neg_47_Left\HR')
parser.add_argument('--nx', type=int, default=494)
parser.add_argument('--ny', type=int, default=614)
parser.add_argument('--nz', type=int, default=752)
args = parser.parse_args()


def extract_dataset():
    """
    将path所指文件划分为训练集和测试集
    :param path:
    :return:
    """
    path = args.path
    print(args.path)

    # 将一个dataset下所有文件
    folder_suffix = ['HR', 'LR\X2']
    supported_img_formats = (".DAT")

    for idx_suffix in range(2):
        path_orginal = os.path.join(path, folder_suffix[idx_suffix])
        path_train = os.path.join(path+'_train', folder_suffix[idx_suffix])
        path_test = os.path.join(path+'_test', folder_suffix[idx_suffix])
        os.makedirs(path_train, exist_ok=True)
        os.makedirs(path_test, exist_ok=True)
        print(path_orginal)
        print(path_train)
        print(path_test)

        for filename in os.listdir(path_orginal):
            print(filename)
            if not filename.endswith(supported_img_formats):
                continue

            dataset = np.fromfile(os.path.join(path_orginal, filename), dtype=np.uint8)
            dataset = dataset.reshape(int(nx/(idx_suffix+1)), int(ny/(idx_suffix+1)), nz)
            trainset = dataset[:, :, 400:nz]
            testset = dataset[:, :, 0:400]
            print(f"dataset shape : {np.shape(dataset)}")
            print(f"trainset shape : {np.shape(trainset)}")
            print(f"testset shape : {np.shape(testset)}")

            trainset.tofile(os.path.join(path_train, filename))
            testset.tofile(os.path.join(path_test, filename))

def extract_file():
    path = args.path
    filename = os.path.splitext(os.path.basename(path))[0]
    # 获取三维
    nx, ny, nz = get_3d(filename)
    # 加载文件
    file = np.fromfile(path, dtype=np.uint8)
    file = file.reshape(nx, ny, nz)
    print(f"\noriginal : {file.shape}")
    train = file[..., 400:nz]
    test = file[..., 0:400]
    print(f"train : {train.shape}")
    print(f"test : {test.shape}")
    p1, p2 = os.path.splitext(path)
    train.tofile(p1+'_train'+p2)
    test.tofile(p1+'_test'+p2)




if __name__ == '__main__':
    args.path = r'D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\3D\HR\Neg_07_Left.DAT'
    extract_file()
