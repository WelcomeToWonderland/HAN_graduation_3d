import os
import argparse
import cv2
import numpy as np

# parse args
parser = argparse.ArgumentParser(description='extract testset from dataset form 300 ~ 499')
parser.add_argument('--path', type=str, default=r'D:\workspace\dataset\OABreast\clipping\Neg_47_Left\HR')
parser.add_argument('--path_testset', type=str,
                    default=r'D:\workspace\dataset\OABreast\clipping\testset\Neg_47_Left\HR')
parser.add_argument('--nx', type=int, default=494)
parser.add_argument('--ny', type=int, default=614)
parser.add_argument('--nz', type=int, default=752)
args = parser.parse_args()


def extract():
    path = args.path
    path_testset = args.path_testset
    nx = int(args.nx)
    ny = int(args.ny)
    nz = args.nz

    print(args.path)

    # create testset dirs
    os.makedirs(path_testset, exist_ok=True)

    supported_img_formats = (".DAT")

    # Downsample HR images
    for filename in os.listdir(path):

        print(filename)

        if not filename.endswith(supported_img_formats):
            continue

        print(os.path.join(path, filename))

        dataset = np.fromfile(os.path.join(path, filename), dtype=np.uint8)

        dataset = dataset.reshape(nx, ny, nz)
        testset = np.zeros((nx, ny, 200), dtype=np.uint8)
        print(f"dataset shape : {np.shape(dataset)}")
        print(f"testset shape : {np.shape(testset)}")
        for idx_dataset, idx_testset in zip(range(300, 500), range(200)):
            print(f"{idx_dataset}")
            print(f"{idx_testset}")
            testset[:, :, idx_testset] = dataset[:, :, idx_dataset]

        testset.tofile(os.path.join(path_testset, filename))


if __name__ == '__main__':
    extract()
