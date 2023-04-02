import torch.utils.data as data
import os
import numpy as np
from data import common
import random

class OABreast(data.Dataset):
    # 函数组-1
    def __init__(self, args, name='', train=True, benchmark=False):
        print('Making dataset oabreast_test...')
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

        # repeat 作用有待探明
        if train:
            n_patches = args.batch_size * args.test_every
            # 待修改
            # n_images = len(args.data_train) * len(self.images_hr)
            n_images = len(args.data_train) * np.shape(self.images_hr)[2]
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def _set_filesystem(self, dir_data):
        '''
        去掉了input_large相关语句（benchmark中也去掉了）
        :param dir_data:
        :return:
        '''
        self.apath = os.path.join(dir_data, self.name)
        scale_dir = f'X{self.scale[0]}'
        self.dir_lr = os.path.join(dir_data, self.name, 'LR',scale_dir)
        self.dir_hr = os.path.join(dir_data, self.name, 'HR')
        self.ext = '.DAT'

    def _scan(self):
        '''
        读取dat文件，加载所有图片信息，放入list
        :return:
        '''
        list_hr = []
        list_lr = [[] for _ in self.scale]

        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]
            list_hr = np.fromfile(os.path.join(self.dir_hr, filename + self.ext), dtype=np.uint8)
            list_hr = list_hr.reshape(self.args.nx_test, self.args.ny_test, self.args.nz_test)
        for entry in os.scandir(self.dir_lr):
            filename = os.path.splitext(entry.name)[0]
            for si, s in enumerate(self.scale):
                list_lr[si] = np.fromfile(os.path.join(self.dir_lr, filename + self.ext), dtype=np.uint8)
                list_lr[si] = list_lr[si].reshape(int(self.args.nx_test / s), int(self.args.ny_test / s), self.args.nz_test)

        return list_hr, list_lr

    # 函数组-2
    def __getitem__(self, idx):
        '''
        函数修改，不再返回filename，改成返回idx
        :param idx:
        :return:
        '''
        lr, hr = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t[0], pair_t[1], idx

    def _load_file(self, idx):
        '''
        函数修改：不再是加载图片（已经加载在list中），而是将图片从list中取出
        :param idx:
        :return:
        '''
        idx = self._get_index(idx)
        hr = self.images_hr[:, :, idx]
        lr = self.images_lr[self.idx_scale][:, :, idx]

        return lr, hr

    def _get_index(self, idx):
        if self.train:
            return idx % np.shape(self.images_hr)[2]
        else:
            return idx

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr, hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            #print(hr.shape)
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            # 校对hr与lr大小（成scale倍数）
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    # 函数-3
    def __len__(se1f):
        if se1f.train:
            return np.shape(se1f.images_hr)[2] * se1f.repeat
        else:
            return np.shape(se1f.images_hr)[2]

    # 函数-4
    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

