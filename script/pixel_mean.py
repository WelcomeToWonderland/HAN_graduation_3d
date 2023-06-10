from scipy import io
import os
import torch
import numpy as np

def pixel_mean(path, key, pixel_range, save_foldername):
    """
    计算像素值均值
    :param path: 文件夹路径
    :param key: 读取mat文件，获得字典，filed是想要获取数据，在字典中对应的键
    :return: 
    """
    # 建立输出文件夹
    save_path = os.path.join(r'../script_results', save_foldername)
    os.makedirs(save_path, exist_ok=True)
    # 建立日志
    log_file = open(os.path.join(save_path, 'log.txt'), 'w')
    # 读取数据
    sum = 0
    num = 0
    for filename in os.listdir(path):
        temp = os.path.join(path, filename)
        file = io.loadmat(temp)
        data = file[key].astype(np.float32)
        data = data / pixel_range
        sum = sum + np.sum(data)
        num = num + data.size
    mean = sum / num
    print(f"mean: {mean}")
    log_file.write(f"mean: {mean}\n")
    #

if __name__ == '__main__':
    paths = [r'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_low_3\HR',
             r'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_low_3\HR_train',
             r'D:\workspace\dataset\USCT\clipping\pixel_translation\bicubic_3d_float_other_low_3\HR_test']
    key = 'f1'
    pixel_range = 1000
    save_foldernames = [r'mean_bicubic_3d_float_other_low_3_HR_all',
                        r'mean_bicubic_3d_float_other_low_3_HR_train',
                        r'mean_bicubic_3d_float_other_low_3_HR_test',]
    for path, save_foldername in zip(paths, save_foldernames):
        pixel_mean(path, key, pixel_range, save_foldername)


