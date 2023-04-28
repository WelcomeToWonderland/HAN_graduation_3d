import os
import glob
import random
import pickle

from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data
import pdb
#import pdb

# dataset源文件
class SRData(data.Dataset):
    # 函数组-1
    def __init__(self, args, name='', train=True, benchmark=False):
        self.input_large = (args.model == 'VDSR')

        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data)

        """
        读取文件的两种方式之一：以二进制的方式读取文件
        为二进制文件创建文件夹
        """
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        # 不采用以下方式读取文件
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                self._check_and_load(args.ext, h, b, verbose=True) 
            for i, ll in enumerate(list_lr):
                for l in ll:
                    #pdb.set_trace()
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)

    def _set_filesystem(self, dir_data):
        '''
        结合dir_data和数据集name，拼接成hr和lr文件夹路径
        :param dir_data:
        :return:
        '''
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        """
        hr后缀与lr后缀
        """
        self.ext = ('.png', '.png')

    def _scan(self):
        '''
        获取hr和lr的所有图片的完整路径列表
        :return:
        '''
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        """
        为了lr与hr一一对应，直接根据hr文件名，生成对应scale的lr文件名
        """
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename,_ = os.path.splitext( os.path.basename(f) )[0].split('_')
            for si, s in enumerate(self.scale):
                # 图片完整路径
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}{}{}'.format(
                        s, filename, '_LR', self.ext[1]
                    )
                ))

        return names_hr, names_lr

    # 函数组-2
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename

    def _load_file(self, idx):
        '''
        根据list中的路径，加载图片
        :param idx:
        :return:
        '''
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]
        #print('！！!！!！!!！',f_lr)
        #pdb.set_trace()
        """
        f_hr : 图片完整路径
        filename : 图片文件名称（不包含文件后缀）
        返回的是文件名称而不是文件路径
        """
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def _get_index(self, idx):
        # if self.train:
        #     return idx % len(self.images_hr)
        # else:
        #     return idx
        return idx

    # 函数-3
    def __len__(self):
        return len(self.images_hr)

    # 未分类函数
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

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
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

