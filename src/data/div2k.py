import os
import srdata
import glob

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        """
        处理数据范围
        :param args:
        :param name:
        :param train:
        :param benchmark:
        """
        print('Making dataset div2k...')
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        """
        仅获取文件名列表
        :return:
        """
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        """
        为了lr与hr一一对应，直接根据hr文件名，生成对应scale的lr文件名
        """
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename = os.path.splitext(os.path.basename(f))[0]
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, 'X{}/{}x{}{}'.format(
                        s, filename, s, self.ext[1]
                    )
                ))
        """
        超出边界最大值时，取边界最大值，取所有元素
        """
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(DIV2K, self)._set_filesystem(dir_data)
        # scale_dir = f'X{self.scale[0]}'
        # self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic', scale_dir)
        """
        scale_dir = f'X{self.scale[0]}'
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic', scale_dir)
        不需要添加最后的scale_dir，srdata中的_scan函数，遍历scale，添加Xscanle后缀
        """
        self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')

        # if self.input_large: self.dir_lr += 'L'

