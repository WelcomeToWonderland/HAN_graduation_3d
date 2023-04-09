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

    """
        if args.template.find('HAN') >= 0:
        args.model = 'HAN'
        args.n_resgroups = 10
        args.n_resblocks = 20
        args.n_feats = 64
        args.chop = True
    """
    # 预训练模型地址
    path_pretain = r'D:\workspace\pre-trained-model\HAN\three_channel\HAN_BIX2.pt'
    path2save = r'D:\workspace\pre-trained-model\HAN\single_channel\HAN_BIX2.pt'
    # 建立白板模型
    module = import_module('model.han')
    han = module.make_model(args)
    # 加载预训练模型参数
    state_dict = torch.load(path_pretain)
    han.load_state_dict(state_dict, strict=False)
    # 修改模型的head
    print(han.head[0])
    weight = han.head[0].weight.data
    weight_subset = weight[:, 0:1, :, :]
    modules_head = nn.Conv2d(1, 64, 3, padding=(3//2), bias=True)
    modules_head.weight.data = weight_subset
    modules_head = [modules_head]
    head = nn.Sequential(*modules_head)
    han.head = head
    print(han.head)
    # 保存模型状态字典（head部分是白板）
    torch.save(han.state_dict(), path2save)