import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm
import numpy as np
import pdb
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model, ckp)

        # 加载load中的优化器参数
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        #print(self.optimizer.get_last_epoch())
        epoch = self.optimizer.get_last_epoch() + 1
        #pdb.set_trace
        lr = self.optimizer.get_lr()
        """
        往log.txt中写入内容
        """
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        """
        开启train模式
        """
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        """
        TEMP
        
        如何实现多scale处理，还是不清楚
        """
        self.loader_train.dataset.set_scale(0)

        """
        enumerate(self.loader_train)
        ？调用dataset的getitem()
        """
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            """
            0 : idx_scale
            HAN模型没有set_scale函数，没有用
            """
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            """
            反向传播
            计算损失函数对于模型参数的梯度
            """
            loss.backward()
            """
            梯度剪裁，防止梯度爆炸
            将梯度限制在阈值self.args.gclip范围内
            """
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            """
            print_every(_batch)
            """
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    # 出现错误，应该是三维像素矩阵的第三维，而不是所有像素点的数目
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        #print(epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()

        # test中才有 保存 操作
        if self.args.save_results: self.ckp.begin_background()

        # 获取dataset
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                # oabreast_2d
                if self.args.is_2d:
                    """
                    testset中只存放了一份数据
                    根据data_test，确定三维
                    """
                    nx, ny, nz = utility.get_3d(self.args.data_test[0])
                    sr_dat = np.zeros((nx, ny, nz), dtype=np.uint8)
                # psnr、ssim数据记录
                calc_psnr_mean = 0
                psnr_mean = 0
                ssim_mean = 0
                # 计数
                num = 0
                # 从dataset中，获取图像
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    """
                    将连续数据，离散化到rgb_range的范围，因此可以进行存储
                    """
                    sr = utility.quantize(sr, self.args.rgb_range)

                    """
                    为什么是filename[0]，而不是filename
                    dataset的__getitem__函数，返回的是字符串，而不是列表
                    进行测试
                    print(f"type(filename) : {type(filename)}")
                    print(f"type(filename[0]) : {type(filename[0])}")
                    print(f"filename : {filename}")
                    print(f"filename[0] : {filename[0]}")
                    """
                    if self.args.save_results:
                        if self.args.is_2d:
                            sr_dat[:, :, filename[0]] = sr.cpu().numpy()[0, 0, :, :]
                        else:
                            save_list = [sr]
                            if self.args.save_gt:
                                save_list.extend([lr, hr])
                            self.ckp.save_results(d, filename[0], save_list, scale)

                    # png图片dataset保存输出
                    # save_list = [sr]
                    # if self.args.save_gt:
                    #     save_list.extend([lr, hr])
                    # if self.args.save_results:
                    #     self.ckp.save_results(d, filename[0], save_list, scale)

                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    #tensorboard
                    num += 1
                    calc_psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    calc_psnr_mean += calc_psnr
                    # sr = sr.cpu().numpy()[0, 0, :, :].astype(np.uint8)
                    # hr = hr.cpu().numpy()[0, 0, :, :].astype(np.uint8)
                    # psnr = peak_signal_noise_ratio(hr, sr, data_range=5)
                    # print(f"psnr: {psnr}")
                    # psnr_mean += psnr
                    # ssim = structural_similarity(hr, sr)
                    # print(f"ssim: {ssim}")
                    # ssim_mean += ssim
                    # self.ckp.writer.add_scalar(r'calc_psnr', calc_psnr, (epoch+1)*len(d) + num)
                    # self.ckp.writer.add_scalar(r'psnr', psnr.item(), (epoch+1)*len(d) + num)
                    # self.ckp.writer.add_scalar(r'ssim', ssim.item(), (epoch+1)*len(d) + num)
                # tensorboard
                calc_psnr_mean /= len(d)
                # psnr_mean /= len(d)
                # ssim_mean /= len(d)
                self.ckp.writer.add_scalar(r'calc_psnr_mean', calc_psnr_mean, epoch + 1)
                # self.ckp.writer.add_scalar(r'psnr_mean', psnr_mean.item(), epoch + 1)
                # self.ckp.writer.add_scalar(r'ssim_mean', ssim_mean.item(), epoch + 1)
                if self.args.save_results:
                    if self.args.is_2d:
                        self.ckp.save_results_dat(d, sr_dat, scale)
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale]
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        # 保存test效果最好的模型
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        """
        将输入张量，改变精度，映射到计算设备上
        以列表的形式返回所有输入张量的处理结果
        :param args:
        :return:
        """
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        '''
        决定是否结束train或者test
        test：执行一次
        train：执行eporch次
        :return:
        '''
        if self.args.test_only:
            # test_only:true时，在此处执行模型test
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch > self.args.epochs

