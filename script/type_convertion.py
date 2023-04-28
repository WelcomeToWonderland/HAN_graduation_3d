import argparse
import os
from scipy import io
import numpy as np
# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=r'',
                    help='')
args = parser.parse_args()

def convert(target):
    """
    给出
    path：dataset文件夹路径
    自动将dataset的HR mat文件，转换为tatget类型数据
    """
    print(f"\ntarget : {target}")
    path = args.path
    # 判断path是否合理
    if os.path.isdir(path):
        print(f"datasset dir : {args.path}")
    else:
        print(f"path:{path} doesn`t exist, process terminates")
    # 获取hr dir
    hr_dir = os.path.join(args.path, 'HR')
    print(f"hr dir : {hr_dir}")
    # 遍历获取文件
    for filename in os.listdir(hr_dir):
        print(f"filename : {filename}")
        path_temp = os.path.join(hr_dir, filename)
        # 获取像素矩阵
        file = io.loadmat(path_temp)
        data = file['f1']
        data = data.astype(target)
        file['f1'] = data
        io.savemat(path_temp, file)

if __name__ == '__main__':
    target = np.uint16
    # 2d
    path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\2d'
    for item in os.listdir(path):
        args.path = os.path.join(path, item)
        convert(target)
    # 3d
    args.path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d'
    convert(target)