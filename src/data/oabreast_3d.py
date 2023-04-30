import torch.utils.data as data
import os
import numpy as np
from data import common
import random
import glob
from utility import get_3d

class OABreast(data.Dataset):
    # 函数组-1
    def __init__(self, args, name='', train=True, benchmark=False):
        print('Making dataset oabreast 3d...')
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

        """
        self.repeat
        在__len__函数中，有使用
        """
        if train:
            """
            test_every(_batch)
            n_patches 两次test之间，使用了n_patches张图像（patch），进行train
            """
            n_patches = args.batch_size * args.test_every
            args.test_every = 10

            """
            修改前
            n_images = len(args.data_train) * len(self.images_hr)
                        
            n_images理解为，所有数据集中图像数量之和（单个数据集，就是这个数据集中）
            
            疑惑
            数据集个数*本数据集中数据数量，但是不同数据集中的数据数量是不同的
            那这样的话，不同数据集对应dataset类求出的n_images是不同的，求出的repeat也是不同的
            """
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                """
                self.repeat
                两次test之间，同一张图像需要重复多少次
                """
                self.repeat = max(n_patches // n_images, 1)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        # scale_dir = f'X{self.scale[0]}'
        self.dir_lr = os.path.join(dir_data, self.name, 'LR')
        self.dir_hr = os.path.join(dir_data, self.name, 'HR')
        self.ext = '.DAT'

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

    # 函数组-2
    def __getitem__(self, idx):
        '''
        按顺序返回：lr， hr， filename
        :param idx:
        :return:
        '''
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        # pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range, is_3d=self.args.is_3d)
        return pair_t[0], pair_t[1], filename

    def _load_file(self, idx):
        '''
        加载图片
        :param idx:
        :return:
        '''
        # 获取文件名
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        # 确定图片三维
        filename = os.path.splitext(os.path.basename(f_hr))[0]
        nx, ny, nz = get_3d(filename)
        # 加载文件
        scale = self.scale[self.idx_scale]
        hr = np.fromfile(f_hr, dtype=np.uint8)
        hr = hr.reshape(nx, ny, nz)
        lr = np.fromfile(f_lr, dtype=np.uint8)
        lr = lr.reshape(nx//scale, ny//scale, nz//scale)
        # 增加通道维度
        lr = np.expand_dims(lr, axis=3) if lr.ndim == 3 else lr
        hr = np.expand_dims(hr, axis=3) if hr.ndim == 3 else hr
        # 返回文件，以及文件名称
        return lr, hr, filename

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
            ih, iw, id = lr.shape[:3]
            hr = hr[0:ih * scale, 0:iw * scale, 0:id*scale]

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
