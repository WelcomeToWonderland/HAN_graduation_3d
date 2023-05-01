import numpy as np
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi import mlab
from src.utility import get_3d
import os

def matplot(path):
    if path.endswith('.mat'):
        file = io.loadmat(path)
        data = file['f1']
    if path.endswith('.DAT'):
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
    plt.show()
    plt.close()

def mlab_plot(path):
    file = io.loadmat(path)
    data = file['f1']
    # 创建一个 3D 图形
    fig = mlab.figure()
    # 绘制 3D 点图
    shape = data.shape
    x, y, z = np.indices((shape[0], shape[1], shape[2]))
    mlab.points3d(x, y, z, data.flatten(), mode='cube', scale_factor=1)
    mlab.show()


path_original = r'D:\workspace\dataset\USCT\original\HR\20220517T112745.mat'
path_translation = r'D:\workspace\dataset\USCT\clipping\pixel_translation\every_other_points_3d\USCT_3d_train\HR\20220517T112745.mat'
path_dat = r''

matplot(path_original)
# mlab_plot(path_original)