import os
from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.parallel as P
import torch.utils.model_zoo
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

class Model(nn.Module):
    # 函数组1
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Making model...')

        self.input_large = (args.model == 'VDSR')

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models

        """
        forward中使用
        """
        self.is_3d = args.is_3d

        if args.model.lower() == 'han':
            if args.is_3d:
                module = import_module('model.han_3d')
            else:
                module = import_module('model.han')
        else:
            module = import_module('model.' + args.model.lower())

        """
        Modle对象有model属性
        Model对象有forword函数
        Model.modle也有forword函数
        """
        self.model = module.make_model(args).to(self.device)
        if args.precision == 'half':
            self.model.half()

        """
        load函数一定会执行
        """
        self.load(
            ckp.get_path('model'),
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )
        """
        将self.model，输出到ckp.log_file，也就是log.txt
        log.txt记录模型结构，和训练测试过程中产生的信息
        
        config.txt记录args
        """
        print(self.model, file=ckp.log_file)

    def load(self, apath, pre_train='', resume=-1, cpu=False):
        """
        option.py中，resume默认0
        这里的resume默认-1，作用不大
        """
        load_from = None
        kwargs = {}
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}

        if resume == -1:
            load_from = torch.load(
                os.path.join(apath, 'model_latest.pt'),
                **kwargs
            )
        elif resume == 0:
            if pre_train == 'download':
                print('Download the model')
                dir_model = os.path.join('..', 'models')
                os.makedirs(dir_model, exist_ok=True)
                load_from = torch.utils.model_zoo.load_url(
                    self.model.url,
                    model_dir=dir_model,
                    **kwargs
                )
            elif pre_train:
                """
                重新开始训练的时候
                resume=0
                pre_train=’‘
                这样才不会报错
                """
                print('Load the model from {}'.format(pre_train))
                load_from = torch.load(pre_train, **kwargs)
        else:
            load_from = torch.load(
                os.path.join(apath, 'model_{}.pt'.format(resume)),
                **kwargs
            )

        if load_from:
            self.model.load_state_dict(load_from, strict=False)

    # 函数组2
    def forward(self, x, idx_scale):
        self.idx_scale = idx_scale
        """
        han模型没有set_scale函数
        """
        if hasattr(self.model, 'set_scale'):
            self.model.set_scale(idx_scale)

        """
        self.training在这个文件中第一次出现
        ？在调用处指明
        """
        if self.training:
            if self.n_GPUs > 1:
                """
                多gpu处理
                
                loss模块中也有对应的多gpu处理
                """
                return P.data_parallel(self.model, x, range(self.n_GPUs))
            else:
                """
                调用model.forward函数
                """
                return self.model(x)
        else:
            """
            HAN　template　默认
            chop = Ture
            self_emsemble = Fasle
            """
            if self.is_3d:
                if self.chop:
                    forward_function = self.forward_chop_3d
                else:
                    forward_function = self.model.forward

                if self.self_ensemble:
                    return self.forward_x8(x, forward_function=forward_function)
                else:
                    return forward_function(x)
            else:
                if self.chop:
                    forward_function = self.forward_chop
                else:
                    forward_function = self.model.forward

                if self.self_ensemble:
                    return self.forward_x8(x, forward_function=forward_function)
                else:
                    return forward_function(x)

    def forward_chop(self, x, shave=10, min_size=160000):
        """
        像素块剪裁
        为了加快运算，将lr划分成4块
        对分块进行超分计算
        最后拼接得到完整sr
        """
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 4)
        # 剪裁chop处理
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        """
        并不是将x划分成不重不漏的四块
        不重不漏应该是0:h_size & h_size:h
        这里确保的是，每块大小都是w_size * h_size（如果h和w是偶数，那就是不重不漏，考虑到了奇数的情况）：0:h_size & (h - h_size):h
        每个块都比四分之一的lr更大，因此不用担心h是奇数，2*h_half<h，导致无法拼出完整sr
        """
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
        # 依照分块大小，分别处理lr分块，得到sr分块
        if w_size * h_size < min_size:
            """
            分块比较小
            一块gpu处理一个分块
            """
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            """
            分块比较大
            再次调用自身forward_chop，将分块再次划分
            """
            sr_list = [
                self.forward_chop(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]
        # 拼接sr分块，得到完整sr
        """
        因为
        """
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        return output

    def forward_chop_3d(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = min(self.n_GPUs, 8)

        b, c, h, w, d = x.size()
        h_half, w_half, d_half = h // 2, w // 2, d // 2
        h_size, w_size, d_size = h_half + shave, w_half + shave, d_half + shave
        """
        分成八块
        """
        lr_list = [
            x[:, :, 0:h_size, 0:w_size, 0:d_size],
            x[:, :, 0:h_size, (w - w_size):w, 0:d_size],
            x[:, :, (h - h_size):h, 0:w_size, 0:d_size],
            x[:, :, (h - h_size):h, (w - w_size):w, 0:d_size],
            x[:, :, 0:h_size, 0:w_size, (d-d_size):d],
            x[:, :, 0:h_size, (w - w_size):w, (d-d_size):d],
            x[:, :, (h - h_size):h, 0:w_size, (d-d_size):d],
            x[:, :, (h - h_size):h, (w - w_size):w, (d-d_size):d]
        ]

        if w_size * h_size * d_size < min_size:
            sr_list = []
            for i in range(0, 8, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [
                self.forward_chop_3d(patch, shave=shave, min_size=min_size) \
                for patch in lr_list
            ]

        h, w, d = scale * h, scale * w, scale * d
        h_half, w_half, d_half = scale * h_half, scale * w_half, scale * d_half
        h_size, w_size, d_size = scale * h_size, scale * w_size, scale * d_size
        # shave *= scale

        output = x.new(b, c, h, w, d)
        output[:, :, 0:h_half, 0:w_half, 0:d_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half, 0:d_half]
        output[:, :, 0:h_half, w_half:w, 0:d_half] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size, 0:d_half]
        output[:, :, h_half:h, 0:w_half, 0:d_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half, 0:d_half]
        output[:, :, h_half:h, w_half:w, 0:d_half] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size, 0:d_half]

        output[:, :, 0:h_half, 0:w_half, d_half:d] \
            = sr_list[4][:, :, 0:h_half, 0:w_half, (d_size - d + d_half):d_size]
        output[:, :, 0:h_half, w_half:w, d_half:d] \
            = sr_list[5][:, :, 0:h_half, (w_size - w + w_half):w_size, (d_size - d + d_half):d_size]
        output[:, :, h_half:h, 0:w_half, d_half:d] \
            = sr_list[6][:, :, (h_size - h + h_half):h_size, 0:w_half, (d_size - d + d_half):d_size]
        output[:, :, h_half:h, w_half:w, d_half:d] \
            = sr_list[7][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size, (d_size - d + d_half):d_size]

        return output

    def forward_x8(self, *args, forward_function=None):
        """
        进行八种变换，增强test数据
        """
        def _transform(v, op):
            """
            进行三种变化
            累计有2*2*2=8种结果
            """
            if self.precision != 'single': v = v.float()
            # 转换到ndarray
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                if v2np.ndim == 4:
                    tfnp = v2np.transpose((0, 1, 3, 2)).copy()
                else:
                    tfnp = v2np.transpose((0, 1, 4, 3, 2)).copy()
            # 转换到tensor，且转移到计算设备
            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        # 变换test lr
        list_x = []
        """
        实际上args只有一个元素：lr
        """
        for a in args:
            x = [a]
            """
            1
            2
            4
            8
            """
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])
            """
            list_x 双重list
            """
            list_x.append(x)

        # 计算 test sr
        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

    # 函数组3
    def save(self, apath, epoch, is_best=False):
        """
        checkpoint的save函数调用，统一保存各种信息
        视情况，最多保存三种类型的模型参数文件
        :param apath:
        :param epoch:
        :param is_best:
        :return:
        """
        save_dirs = [os.path.join(apath, 'model_latest.pt')]

        if is_best:
            save_dirs.append(os.path.join(apath, 'model_best.pt'))
        if self.save_models:
            save_dirs.append(
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

        for s in save_dirs:
            torch.save(self.model.state_dict(), s)


