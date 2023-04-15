import os
import argparse
import shutil

# parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', type=str, default=r'',
                    help='总文件夹')
args = parser.parse_args()

def establish():
    """
    提供文件夹路径，文件夹存放hr文件
    1、3d:在文件夹下建立HR文件夹，将所有文件复制到HR文件夹下
    2、2d:为每个文件，以文件名称，建立对应文件夹；在对应文件夹下，建立HR文件夹，将文件放入HR文件夹下，删除原文件
    :return:
    """
    data_dir = args.data_dir
    supported_ext = '.mat'
    # 检验文件夹路径合法性
    if not os.path.isdir(data_dir):
        print(f"data_dir doesn`t exist, function terminates")
        return
    filenames = os.listdir(data_dir)
    if len(filenames) == 0:
        print(f"no files, function terminates")
        return
    print(f"\ndata dir : {data_dir}")
    # 2d
    for filename in filenames:
        if filename.endswith(supported_ext):
            basename, _ = os.path.splitext(filename)
            hr_dir = os.path.join(data_dir, basename, 'HR')
            os.makedirs(hr_dir, exist_ok=True)
            """
            src源文件路径名
            dst目标文件夹路径名
            """
            print(f'copy file ：{filename}')
            shutil.copy(os.path.join(data_dir, filename), hr_dir)
    # 3d
    hr_dir = os.path.join(data_dir, 'HR')
    os.makedirs(hr_dir, exist_ok=True)
    for filename in filenames:
        if filename.endswith(supported_ext):
            """
            src源文件路径名
            dst目标文件路径名
            """
            print(f'move file ：{filename}')
            shutil.move(os.path.join(data_dir, filename), os.path.join(hr_dir, filename))

def get_dataset_names():
    str = ''
    for name in os.listdir(args.data_dir):
        if name != 'HR':
            str = str + '+' + name
    if str[0] == '+':
        str = str[1:]
    return str

if __name__ == '__main__':
    args.data_dir = r'D:\workspace\dataset\USCT\clipping'
    # establish()
    print(get_dataset_names())