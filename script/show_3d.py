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

# def matplot_2d(path, savename):
#     if path.endswith('.mat'):
#         file = io.loadmat(path)
#         data = file['f1']
#     elif path.endswith('.DAT'):
#         filename = os.path.basename(path)
#         basename, _ = os.path.splitext(filename)
#         x, y, z = get_3d(basename)
#         data = np.fromfile(path)
#         data = data.reshape(x, y, z)
#     # 创建一个 3D 图形
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     # 绘制 3D 点图
#     shape = data.shape
#     x, y, z = np.indices((shape[0], shape[1], shape[2]))
#     ax.scatter(x, y, z, c=data.flatten())
#     # ax.scatter3D(x, y, z, c=data.flatten())
#     # 展示
#     # plt.show()
#     # 保存文件
#     os.makedirs(os.path.join('.', 'USCT_3d'), exist_ok=True)
#     path_save = os.path.join('.','USCT_3d', savename)
#     print(f"matplot path_save : {path_save}")
#     plt.savefig(path_save, dpi=600)
#     # 关闭
#     plt.close()

def matplot_2d_imshow(path, savefolder, savename):
    # 加载数据
    if path.endswith('.mat'):
        file = io.loadmat(path)
        data = file['f1']
    elif path.endswith('.DAT'):
        filename = os.path.basename(path)
        basename, _ = os.path.splitext(filename)
        x, y, z = get_3d(basename)
        data = np.fromfile(path)
        data = data.reshape(x, y, z)

    # 绘制平面图像
    os.makedirs(os.path.join('../script_results', savefolder), exist_ok=True)
    shape = data.shape
    for pos in range(3):
        if pos == 0:
            temp = data[shape[0]//2]
        elif pos == 1:
            temp = data[:, shape[1]//2]
        elif pos == 2:
            temp = data[..., shape[2]//2]
        plt.imshow(temp)
        plt.colorbar()
        # plt.show()
        basename, ext = os.path.splitext(savename)
        temp_savename = basename + f'_{pos}' + ext
        path_save = os.path.join('.', savefolder, temp_savename)
        print(f"matplot path_save : {path_save}")
        plt.savefig(path_save, dpi=600)
        plt.close()




if __name__ == '__main__':
    # path_original = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d\HR'
    # for filename in os.listdir(path_original):
    #     path = os.path.join(path_original, filename)
    #     basename, _ = os.path.splitext(filename)
    #     savename = basename + '.png'
    #     savefolder = 'USCT_2d_inshow_translation'
    #     matplot_2d_imshow(path, savefolder, savename)

    path_processed = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d_to_oabreast\HR'
    for filename in os.listdir(path_processed):
        path = os.path.join(path_processed, filename)
        basename, _ = os.path.splitext(filename)
        savename = basename + '.png'
        savefolder = 'USCT_2d_inshow_translation_to_oabreast'
        matplot_2d_imshow(path, savefolder, savename)