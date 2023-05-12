import numpy as np
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# from mayavi import mlab
from src.utility import get_3d
import os
from script.common import delete_folder
import argparse
# parse args
parser = argparse.ArgumentParser(description='Downsize images at 2x using bicubic interpolation')
parser.add_argument('--img_type', type=str, default=r'HR',
                    help='')
parser.add_argument('--dimension', type=int, default=2,
                    help='')
args = parser.parse_args()

def matplot_3d(path, savename):
    if path.endswith('.mat'):
        file = io.loadmat(path)
        data = file['f1']
    elif path.endswith('.DAT'):
        filename = os.path.basename(path)
        basename, _ = os.path.splitext(filename)
        x, y, z = get_3d(basename)
        data = np.fromfile(path)
        data = data.reshape(x, y, z)
    # 创建一个 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制 3D 点图
    shape = data.shape
    x, y, z = np.indices((shape[0], shape[1], shape[2]))
    ax.scatter(x, y, z, c=data.flatten())
    # ax.scatter3D(x, y, z, c=data.flatten())
    # 展示
    # plt.show()
    # 保存文件
    os.makedirs(os.path.join('../script_results', '../script_results/USCT_3d'), exist_ok=True)
    path_save = os.path.join('.', '../script_results/USCT_3d', savename)
    print(f"matplot path_save : {path_save}")
    plt.savefig(path_save, dpi=600)
    # 关闭
    plt.close()

def matplot_2d_imshow(path, savefolder, savename):
    # 加载数据
    if path.endswith('.mat'):
        file = io.loadmat(path)
        if args.img_type=='LR':
            data = file['imgout']
        else:
            data = file['f1']
        # data = file['imgout']
    elif path.endswith('.DAT'):
        basename, _ = os.path.splitext(savename)
        x, y, z = get_3d(basename)
        if args.img_type=='LR' and args.img_type == 2:
            x, y, z = x//2, y//2, z
        data = np.fromfile(path, dtype=np.uint8)
        data = data.reshape(x, y, z)
    # 建立输出保存文件夹
    save_dir = os.path.join('../script_results', savefolder)
    os.makedirs(save_dir, exist_ok=True)
    # 绘制平面图像
    shape = data.shape
    for pos in range(3):
        if pos == 0:
            for idx in range(3):
                if idx == 0:
                    temp = data[shape[0] // 4]
                elif idx == 1:
                    temp = data[shape[0] // 2]
                elif idx == 2:
                    temp = data[shape[0] // 4 * 3]
                plt.imshow(temp)
                plt.colorbar()
                basename, ext = os.path.splitext(savename)
                temp_savename = basename + f'_{pos}_{idx}' + ext
                path_save = os.path.join(save_dir, temp_savename)
                print(f"matplot path_save : {path_save}")
                plt.savefig(path_save, dpi=600)
                plt.close()
        elif pos == 1:
            for idx in range(3):
                if idx == 0:
                    temp = data[:, shape[1] // 4]
                elif idx == 1:
                    temp = data[:, shape[1] // 2]
                elif idx == 2:
                    temp = data[:, shape[1] // 4 * 3]
                plt.imshow(temp)
                plt.colorbar()
                basename, ext = os.path.splitext(savename)
                temp_savename = basename + f'_{pos}_{idx}' + ext
                path_save = os.path.join(save_dir, temp_savename)
                print(f"matplot path_save : {path_save}")
                plt.savefig(path_save, dpi=600)
                plt.close()
        elif pos == 2:
            for idx in range(3):
                if idx == 0:
                    temp = data[..., shape[2]//4]
                elif idx == 1:
                    temp = data[..., shape[2]//2]
                elif idx == 2:
                    temp = data[..., shape[2]//4*3]
                plt.imshow(temp)
                plt.colorbar()
                basename, ext = os.path.splitext(savename)
                temp_savename = basename + f'_{pos}_{idx}' + ext
                path_save = os.path.join(save_dir, temp_savename)
                print(f"matplot path_save : {path_save}")
                plt.savefig(path_save, dpi=600)
                plt.close()

if __name__ == '__main__':

    filenames = ['20220511T153240.mat', '20220517T112745.mat',
                 '50525.mat']

    # algorithm = 'bicubic'
    # args.img_type = 'SR'
    # args.dimension = 3
    # suffix = 'other_low_1'
    # # lr = 5
    # # 初始化：删除之前文件夹
    # # savefolder = f'USCT_{args.dimension}d_{args.img_type}_{algorithm}_bn_lr_{lr}'
    # savefolder = f'USCT_{args.dimension}d_{args.img_type}_{algorithm}_{suffix}'
    # # savefolder = f'USCT_{args.dimension}d_{args.img_type}_{algorithm}'
    # delete_folder(os.path.join('..', 'script_results', savefolder))
    # # 拼接文件path
    # path_original = r'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_low\SRSR'
    # filename = '20220517T112745.mat'
    # # filename = '20220517T112745_x2_SR.mat'
    # path = os.path.join(path_original, filename)
    # # 构建图片名
    # basename, _ = os.path.splitext(filename)
    # # savename = basename + '.png'
    # savename = '20220517T112745' + '.png'
    # # 调用函数
    # matplot_2d_imshow(path, savefolder, savename)

    nums = [1, 3, 7, 10]
    for num in nums:
        algorithm = 'HAN'
        args.img_type = 'SR'
        args.dimension = 3

        # 初始化：删除之前文件夹
        savefolder = f'USCT_{args.dimension}d_{args.img_type}_{algorithm}_bn_lr_{num}_other'
        # savefolder = f'USCT_3d_SR_HAN_other_low_{num}'
        delete_folder(os.path.join('..', 'script_results', savefolder))

        # path_original = rf'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_low_{num}\LR\X2'
        # filename = '20220517T112745.mat'
        path_original = rf'/root/autodl-tmp/project/HAN_for_3d/experiment/usct_3d_bn_other_lr_{num}/results-USCT_3d_test'
        filename = '20220517T112745_x2_SR.mat'

        path = os.path.join(path_original, filename)
        basename, _ = os.path.splitext(filename)
        # savename = basename + '.png'
        savename = '20220517T112745' + '.png'
        matplot_2d_imshow(path, savefolder, savename)


    # levels = [0.1, 2, 3, 4, 5]
    # for level in levels:
    #     # 初始换：删除之前文件夹
    #     savefolder = f'USCT_3d_SR_bicubic_other_noise_{level}'
    #     delete_folder(os.path.join('..', 'script_results', savefolder))
    #
    #     for filename in filenames:
    #         path_original = rf'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_noise_{level}\SRSR'
    #         path = os.path.join(path_original, filename)
    #         basename, _ = os.path.splitext(filename)
    #         savename = basename + '.png'
    #         matplot_2d_imshow(path, savefolder, savename)


    # filenames = ['20220510T153337.mat', '20220511T153240.mat', '20220517T112745.mat',
    #              '20220525T153940.mat', '20220526T181025.mat', '20220608T172601.mat',
    #              '20220809T140229.mat', '20220819T162347.mat', '20221114T153716.mat',
    #              '20221116T164200.mat',
    #              '50525.mat', '52748.mat']
    #
    # idx = 1
    #
    # # 初始换：删除之前文件夹
    # savefolder = 'USCT_2d_inshow_translation'
    # delete_folder(os.path.join('..', 'script_results', savefolder))
    # path_original = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d\HR'
    # filename = filenames[idx]
    # path = os.path.join(path_original, filename)
    # basename, _ = os.path.splitext(filename)
    # savename = basename + '.png'
    # matplot_2d_imshow(path, savefolder, savename)
    #
    #
    # savefolder = 'USCT_2d_inshow_translation_to_oabreast'
    # delete_folder(os.path.join('..', 'script_results', savefolder))
    # path_processed = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d_to_oabreast\HR'
    # filename = filenames[idx]
    # path = os.path.join(path_processed, filename)
    # basename, _ = os.path.splitext(filename)
    # savename = basename + '.png'
    # matplot_2d_imshow(path, savefolder, savename)





    # path_original = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d\HR'
    # for filename in os.listdir(path_original):
    #     path = os.path.join(path_original, filename)
    #     basename, _ = os.path.splitext(filename)
    #     savename = basename + '.png'
    #     savefolder = 'USCT_2d_inshow_translation'
    #     matplot_2d_imshow(path, savefolder, savename)
    #
    # path_processed = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d_to_oabreast\HR'
    # for filename in os.listdir(path_processed):
    #     path = os.path.join(path_processed, filename)
    #     basename, _ = os.path.splitext(filename)
    #     savename = basename + '.png'
    #     savefolder = 'USCT_2d_inshow_translation_to_oabreast'
    #     matplot_2d_imshow(path, savefolder, savename)