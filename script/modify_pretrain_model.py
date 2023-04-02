import torch
from src.option import args

# 修改预训练模型的通道数为1

def change_num_of_channel(path):
    None


if __name__ == '__main__':
    path_pretain = f'D:\workspace\pre-trained-model\HAN\single_channel\HAN_BIX2.pt'

    state_dict = torch.load(path_pretain)

    print(state_dict.keys())
