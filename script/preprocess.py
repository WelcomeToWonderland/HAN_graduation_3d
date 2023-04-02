# —*- coding = utf-8 -*-
# @Time : 2023-02-20 18:50
# @Author : 阙祥辉
# @File : preprocess.py
# @Software : PyCharm

import os


def dataset_preprocess(path, flag):
    '''
    删除一个文件夹下特定的文件
    '''
    img_list = os.listdir(path)
    for img in img_list:
        if img.split('.')[0].split('_')[-1] == flag:
            path_ = os.path.join(path, img)
            os.remove(path_)


if __name__ == '__main__':
    path = r'D:\workspace\dataset\Urban100\HR'
    flag = 'LR'
    img_list = os.listdir(path)
    for img in img_list:
        if img.split('.')[0].split('_')[-1] == flag:
            path_ = os.path.join(path, img)
            os.remove(path_)
