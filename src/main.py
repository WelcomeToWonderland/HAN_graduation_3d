import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import gc
from importlib import import_module
import torch.nn as nn

gc.collect()
torch.cuda.empty_cache()
# import sys
# sys.path.append('/root/autodl-tmp/project')


"""
行为：将args.seed设置为cpu和gpu的随机数种子
目的：方便复现
"""
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        # 图片
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    # main()

    # m = import_module('data.oabreast_3d')
    # dataset = getattr(m, 'OABreast')(args, name='OABreast_3d_train')
    # length = len(dataset)
    # print(f"len : {length}")
    # for idx in range(length):
    #     lr, hr, filename = dataset.__getitem__(idx)
    #     print(f"\nhr shape : {hr.shape}")
    #     print(f"lr shape : {lr.shape}")
    #     print(f"filename : {filename}")

    m = import_module('data.usct')
    dataset = getattr(m, 'USCT')(args, name='20220511T153240')
    length = len(dataset)
    print(f"len : {length}")
    for idx in range(length):
        lr, hr, filename = dataset.__getitem__(idx)
        print(f"\nhr shape : {hr.shape}")
        print(f"lr shape : {lr.shape}")
        print(f"filename : {filename}")