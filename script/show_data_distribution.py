import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import os
from src.utility import get_3d
from script.common import delete_folder
import argparse

# parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--mat_field', type=str, default=r'',
                    help='')
args = parser.parse_args()

def show_data_distribution(path, save_folder, more_accuate: bool = True):
    plt.clf()
    print(f"\npath : {path}")
    # 建立输出保存文件夹
    save_dir = os.path.join('../script_results', save_folder)
    delete_folder(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # 建立log文件
    log_file = open(os.path.join(save_dir, 'log.txt'), 'x')
    # 加载数据
    if path.endswith('.mat'):
        file = io.loadmat(path)
        # data = file['f1']
        data = file[args.mat_field]
    elif path.endswith('.DAT'):
        filename = os.path.basename(path)
        basename, _ = os.path.splitext(filename)
        x, y, z = get_3d(basename)
        data = np.fromfile(path)
        data = data.reshape(x, y, z)
    else:
        print(f'the path is wrong, program termnates!')
        return
    # 文件名处理
    filename = os.path.basename(path)
    basename, _ = os.path.splitext(filename)
    log_file.write(f"{filename}\n")
    # 获取data distribution
    hist, bin_edges = np.histogram(data.ravel(), bins=10, range=(data.min(), data.max()))
    print(f"\nhist : {hist}")
    print(f"bin_edges : {bin_edges}")
    log_file.write(f'hist : {hist}\n')
    log_file.write(f'bin_edges : {bin_edges}\n')
    # 绘制直方图
    plt.hist(data.ravel(), bins=10, range=(data.min(), data.max()))
    plt.savefig(os.path.join(save_dir, basename+'_all.png'))
    plt.close()
    # 绘制更加精细的直方图
    if more_accuate:
        plt.hist(data.ravel(), bins=20, range=(1450, 1560))
        plt.savefig(os.path.join(save_dir, basename+'_reduced.png'))
        plt.close()


if __name__ == '__main__':

    args.mat_field = 'img'
    dir_dataset = '/workspace/datasets/OA-breast_correct'
    for foldername in os.listdir(dir_dataset):
        save_folder_prefix = foldername
        resolutions = ['HR', 'LR/X2']
        for resolution in resolutions:
            save_folder = save_folder_prefix + '_' + resolution
            dir_file = os.path.join(dir_dataset, foldername, resolution)
            for filename in os.listdir(dir_file):
                path = os.path.join(dir_file, filename)
                show_data_distribution(path, save_folder=save_folder, more_accuate=False)





    # filenames = ['20220510T153337.mat', '20220511T153240.mat', '20220517T112745.mat',
    #              '20220525T153940.mat', '20220526T181025.mat', '20220608T172601.mat',
    #              '20220809T140229.mat', '20220819T162347.mat', '20221114T153716.mat',
    #              '20221116T164200.mat',
    #              '50525.mat', '52748.mat']
    #
    # idx = 1
    #
    # args.mat_field = 'f1'
    #
    # # 处理前
    # save_folder = 'distribution'
    # # 初始化，删除之前文件
    # delete_folder(os.path.join('../script_results', save_folder))
    # path = r'D:\workspace\dataset\USCT\original\HR'
    # filename = filenames[idx]
    # show_data_distribution(os.path.join(path, filename), save_folder=save_folder)
    #
    # # 处理后
    # save_folder = 'distribution_processed'
    # # 初始化，删除之前文件
    # delete_folder(os.path.join('../script_results', save_folder))
    # path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d_to_oabreast\HR'
    # filename = filenames[idx]
    # show_data_distribution(os.path.join(path, filename), save_folder=save_folder, more_accuate=False)



    # save_folder = 'distribution'
    # delete_folder(os.path.join('../script_results', save_folder))
    # path = r'D:\workspace\dataset\USCT\original\HR'
    # for filename in os.listdir(path):
    #     show_data_distribution(os.path.join(path, filename), save_folder=save_folder)
    #
    # save_folder = 'distribution_processed'
    # delete_folder(os.path.join('../script_results', save_folder))
    # path = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d_to_oabreast\HR'
    # for filename in os.listdir(path):
    #     show_data_distribution(os.path.join(path, filename), save_folder=save_folder, more_accuate=False)