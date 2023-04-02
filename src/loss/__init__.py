import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        # 准备args.loss中的各个loss函数
        self.loss = []
        # 将loss中准备好的loss函数，取出，放入loss_module
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        # loss log，为文件loss.pt做准备
        self.log = torch.Tensor()
        # 创建设备对象，为tensor的计算做准备
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()

        #
        # if not args.cpu and args.n_GPUs > 1:
        #     self.loss_module = nn.DataParallel(
        #         self.loss_module, range(args.n_GPUs)
        #     )
        if not args.cpu and args.n_GPUs > 0:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        # load非空，加载上一次训练的loss参数
        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    # 函数组1
    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        # 加载上次训练的loss权重
        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        # 加载上次训练中每个epoch的loss
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        # 没搞懂
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    # 函数组2
    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    # 函数组3
    def start_log(self):
        # 为什么是1, len(self.loss)：一次记录，有几个loss函数，就有几个对应的loss函数值
        # 默认在第一个维度上，进行拼接
        # 这一个epoch，记录loss做准备
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        # 累加epoch中每个batch的loss值，最后求平均值
        self.log[-1].div_(n_batches)

    # 函数组4
    def display_loss(self, batch):
        """
        获取当前batch的loss平均值，并连接成字符串，字符串的形式返回
        :param batch:
        :return:
        """
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            # c / n_samples：loss当前平均值
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        # 将log中的元素拼接成字符串
        return ''.join(log)

    def plot_loss(self, apath, epoch):
        # 等差数列：start：1   end：epoch   num：epoch
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            # 创建Figure对象
            fig = plt.figure()
            plt.title(label)
            # self.log[:, i] self.loss中第i个loss函数在各个epoch的loss平均值
            # X轴：axis   Y轴：self.log[:, i]
            # self.log[:, i].numpy() 将tensor转ndarray
            # label：线条的标签
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            # legeng：（图表上的）图例，说明
            # 将plt.plot中指定的label，添加进图例中；图例解释了”线条颜色与label“的对应关系
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            # 以pdf的形式形式保存图片
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    # 函数组5
    def forward(self, sr, hr):
        """
        模型的正向传播
        接受输入张量，计算输出张量
        loss的计算
        :param sr:
        :param hr:
        :return:
        """
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                # 累加loss，为之后求平均值做准备
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        # 多个loss函数
        # for l in self.get_loss_module():
        #     if hasattr(l, 'scheduler'):
        #         l.scheduler.step()

        # 单个loss函数
        l = self.get_loss_module()
        if hasattr(l, 'scheduler'):
            l.scheduler.step()





