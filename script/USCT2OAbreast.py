from scipy import io
import argparse
from script.show_3d import matplot
import os
import numpy as np
import time
from script.common import delete_folder

# parse args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--water', type=float, default=1485,
                    help='0')
parser.add_argument('--fibro_glandular_tissue', type=float, default=1490,
                    help='2')
parser.add_argument('--fat', type=float, default=1450,
                    help='3')
parser.add_argument('--skin_layer', type=float, default=1470,
                    help='4')
parser.add_argument('--blood_vessel', type=float, default=1460,
                    help='5')
parser.add_argument('--plot', type=bool, default=False,
                    help='')
parser.add_argument('--reset', type=bool, default=False,
                    help='')
args = parser.parse_args()

def USCT2OAbreast(path_get, path_save, filename):
    """
    数据范围转换
    数据类型转换
    顺带完成数据平移
    :param path:
    :return:
    """
    os.makedirs(path_save, exist_ok=True)
    path_get = os.path.join(path_get, filename)
    path_save = os.path.join(path_save, filename)
    print(f"\npath_get : {path_get}")
    print(f"path_save : {path_save}")
    # 加载数据
    if path_get.endswith('.mat'):
        file = io.loadmat(path_get)
        data = file['f1']
    else:
        print(f'the path_get is wrong, program termnates!')
        return
    # 可视化：处理前
    basename, _ = os.path.splitext(filename)
    savename = basename + '_before.png'
    if args.plot:
        matplot(path_get, savename)
    # 数据转化：遍历所有元素
    x, y, z = data.shape
    for ix in range(x):
        for iy in range(y):
            for iz in range(z):
                if data[ix, iy, iz] + 1e3 < args.fat:
                    data[ix, iy, iz] = 2
                elif data[ix, iy, iz] + 1e3 < args.water:
                    data[ix, iy, iz] = 0
                elif data[ix, iy, iz] + 1e3 < args.fibro_glandular_tissue:
                    data[ix, iy, iz] = 1
                elif data[ix, iy, iz] + 1e3 < args.blood_vessel:
                    data[ix, iy, iz] = 4
                elif data[ix, iy, iz] + 1e3 < args.skin_layer:
                    data[ix, iy, iz] = 3
    data = data.astype(np.uint8)
    file['f1'] = data
    io.savemat(path_save, file)
    # 可视化：处理后
    savename = basename + '_after.png'
    if args.plot:
        matplot(path_save, savename)

if __name__ == '__main__':
    path_get = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d\HR'
    path_save = r'D:\workspace\dataset\USCT\clipping\pixel_translation\3d_to_oabreast\HR'
    filenames = ['20220510T153337.mat', '20220511T153240.mat', '20220517T112745.mat',
                 '20220525T153940.mat', '20220526T181025.mat', '20220608T172601.mat',
                 '20220809T140229.mat', '20220819T162347.mat', '20221114T153716.mat',
                 '20221116T164200.mat',
                 '50525.mat', '52748.mat']

    args.plot = False
    args.reset = False
    if args.reset:
        delete_folder(path_save)
    if args.plot:
        delete_folder(os.path.join('.', '../script_results/USCT_3d'))

    os.makedirs(path_save, exist_ok=True)

    time_begin = time.time()

    filename = filenames[0]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1475.19446776, 1488.59911203, 1502.00375631, 1515.40840059, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[1]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1479.5033464, 1489.84729074, 1500.19123507, 1510.5351794, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[2]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1446.86278048, 1486.25313586, 1525.64349124, 1565.03384662, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[3]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1431.65332707, 1478.8428911, 1526.03245513, 1573.22201916, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    #
    filename = filenames[4]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1394.20524367, 1437.98997211, 1481.77470054, 1525.55942898, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[5]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1457.76418075, 1482.98892203, 1508.21366331, 1533.43840458, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[6]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1490.90121454, 1502.85867445, 1514.81613437, 1526.77359428, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    # 20220819T162347 想要剔除
    # filename = filenames[7]
    # args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
    #     [1495, 1506, 1516, 1531.28691289, 2000]
    # USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[8]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1471.19381085, 1481.57181229, 1491.94981374, 1502.32781518, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[9]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1472.65897464, 1485.03657281, 1497.41417097, 1509.79176913, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[10]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1484.04490116, 1502.80908079, 1521.57326042, 1540.33744004, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    filename = filenames[11]
    args.fat, args.water, args.fibro_glandular_tissue, args.blood_vessel, args.skin_layer = \
        [1471.82995671, 1488.84920356, 1505.86845041, 1522.88769725, 2000]
    USCT2OAbreast(path_get, path_save, filename)

    time_end = time.time()
    print(f"time consuming : {time_end - time_begin}s")