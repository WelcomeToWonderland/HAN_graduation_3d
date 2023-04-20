import torch.utils.data as data
import os
import numpy as np
from data import common
import random
import glob
from src.utility import get_3d
from scipy import io

class USCT(data.Dataset):
    # 函数组-1
    def __init__(self, args, name='', train=True, benchmark=False):
        print('Making dataset usct 3d...')

        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        # 在后续几个函数中有所涉及（commom.get_patch）
        self.input_large = (args.model == 'VDSR')
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0


        self._set_filesystem(args.dir_data)

        list_hr, list_lr = self._scan()
        self.images_hr, self.images_lr = list_hr, list_lr

    def _set_filesystem(self, dir_data):
        '''
        去掉了input_large相关语句（benchmark中也去掉了）

        记录相关路径，但是没有根据路径创建文件
        :param dir_data:
        :return:
        '''
        self.apath = os.path.join(dir_data, self.name)
        # scale_dir = f'X{self.scale[0]}'
        self.dir_lr = os.path.join(dir_data, self.name, 'LR')
        self.dir_hr = os.path.join(dir_data, self.name, 'HR')
        self.ext = '.mat'

    # def _scan(self):
    #     '''
    #     扫描文件夹
    #     读取dat文件，加载所有图片信息，放入list
    #     :return:
    #     '''
    #     list_hr = []
    #     # 读取不同scale的lr文件
    #     list_lr = [[] for _ in self.scale]
    #
    #     for entry in os.scandir(self.dir_hr):
    #         filename = os.path.splitext(entry.name)[0]
    #         list_hr = np.fromfile(os.path.join(self.dir_hr, filename + self.ext), dtype=np.uint8)
    #         list_hr = list_hr.reshape(self.nx, self.ny, self.nz)
    #     for entry in os.scandir(self.dir_lr):
    #         filename = os.path.splitext(entry.name)[0]
    #         for si, s in enumerate(self.scale):
    #             list_lr[si] = np.fromfile(os.path.join(self.dir_lr, filename + self.ext), dtype=np.uint8)
    #             list_lr[si] = list_lr[si].reshape(int(self.nx / s), int(self.ny / s), self.nz)
    #
    #     return list_hr, list_lr

    # 函数组-2

    def _scan(self):
        '''
        扫描文件夹
        获取所有hr和lr文件名
        :return:
        '''
        """
        获取hr文件名
        根据hr文件名，获取不同scale的对应lr文件名
        直接获取文件夹下所有lr文件名，存在hr与lr不配对的风险
        """
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext))
        )
        # 为不同scale，建立对应lr文件名存储list
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            f = os.path.basename(f)
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}'.format(s, f)
                ))
        return names_hr, names_lr

    def __getitem__(self, idx):
        '''
        usct 3d
        返回filename
        :param idx:
        :return:
        '''
        lr, hr, basename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        """
        3d数据，通道数只为1，所以没有设置channel的需求
        set_channel_3d
        """
        # pair = common.set_channel_3d(*pair, n_channels=self.args.n_colors)
        """
        common.np2Tensor：从ndarray到tensor
        trainer.prepare:数据转移到计算设备
        """
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range, is_3d=self.args.is_3d)
        return pair_t[0], pair_t[1], basename

    def _load_file(self, idx):
        '''
        函数修改：不再是加载图片（已经加载在list中），而是将图片从list中取出

        为不存在通道维度的像素矩阵，添加通道维度
        模型，要求数据具有通道维度

        从list中取得filename，加载file，取出data

        :param idx:
        :return:
        '''
        # get filename & basename
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        basename = os.path.splitext(os.path.basename(f_hr))[0]
        # get file & data
        file = io.loadmat(f_hr)
        hr = file['f1']
        file = io.loadmat(f_lr)
        lr = file['imgout']
        # 增加通道维度
        lr = np.expand_dims(lr, axis=3) if lr.ndim == 3 else lr
        hr = np.expand_dims(hr, axis=3) if hr.ndim == 3 else hr
        # 返回文件，以及文件名称
        return lr, hr, basename

    def _get_index(self, idx):
        if self.train:
            # return idx % np.shape(self.images_hr)[2]
            return idx % len(self.images_hr)
        else:
            return idx

    def get_patch(self, lr, hr):
        """
        train:patch和argument
        test:没有patch，只有形状校对，符合成scale倍数就行
        :param lr:
        :param hr:
        :return:
        """
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch_3d(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            #print(hr.shape)
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    # 函数-3
    def __len__(se1f):

        """
        if se1f.train:
            return np.shape(se1f.images_hr)[2] * se1f.repeat
        else:
            return np.shape(se1f.images_hr)[2]

        不再使用args.every_test属性，对应sef.repeat属性
        :return:
        """
        # return np.shape(se1f.images_hr)[2]
        return len(se1f.images_hr)

    # 函数-4
    def set_scale(self, idx_scale):
        if not self.input_large:
            """
            oabreast，只会执行这个分支
            """
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

