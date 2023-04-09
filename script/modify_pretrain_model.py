import torch
from src.option import args
from importlib import import_module


# 修改预训练模型的通道数为1

def change_num_of_channel(path):
    None


if __name__ == '__main__':
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
    # 冻结除最后一层外的全部参数
    for name, param in han.named_parameters():
        print(f"name : {name}")
        # if name not in ["fc.weight", "fc.bias"]:
        #     #         print(name)
        #     param.requires_grad = False  # 关键一步，设置为False之后，优化器溜不会对参数进行更新

