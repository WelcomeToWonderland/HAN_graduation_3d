import os
import argparse
import cv2
import numpy as np

# parse args
parser = argparse.ArgumentParser(description='extract testset from dataset form 300 ~ 499')
parser.add_argument('--path', type=str, default=r'D:\workspace\dataset\OABreast\clipping\Neg_47_Left\HR')
parser.add_argument('--nx', type=int, default=494)
parser.add_argument('--ny', type=int, default=614)
parser.add_argument('--nz', type=int, default=752)
args = parser.parse_args()


def extract():
    path = args.path
    nx = args.nx
    ny = args.ny
    nz = args.nz

    print(args.path)

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


if __name__ == '__main__':
    args.path = r'D:\workspace\dataset\OABreast\clipping\pixel_translation\downing\Neg_07_Left'
    args.nx = 616
    args.ny = 484
    args.nz = 719
    extract()
