from model import common
# import common
import torch
import torch.nn as nn
import pdb

def make_model(args, parent=False):
    return HAN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        """
        global average pooling: feature --> point
        
        nn.AdaptiveAvgPool2d(1)
        自适应平均二维池化，不用指定池化核的大小，指定输出的大小，自动反推池化核大小
        1 是 output size。要求输入二元元组（h，w），仅输入一个数字1，效果就是长宽都是1，（1， 1）
        
        池化操作在通道维度上进行
        在这里，每个通道都有一个计算结果，依据这些结果，计算出每个通道的权重，权重之和为1
        """
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out + x
        out = out.view(m_batchsize, -1, height, width)
        return out

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        """
        python内置函数super，返回父类的临时对象
        super的两个参数：子类名称，子类实例
        调用父类的初始化函数__init__，继承父类的属性和方法（为什么使用方法，不太理解，方法不是在__init__函数中定义）
        """

        modules_body = []
        """
        循环最终效果：卷积+relu+卷积
        """
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        """
        限制
        固定值：act=nn.ReLU(True), res_scale=1
        输入的这两个值无效
        """
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Holistic Attention Network (HAN)
class HAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        """
        rcan : reduction控制特征图的减少
        """
        reduction = args.reduction
        """
        scale是定量
        模型最后的上采样模型是确定的
        ？那怎么做到同一模型，实现不同scale的提升
        """
        scale = args.scale[0]
        act = nn.ReLU(True)


        """
        修改之后，HAN模型默认不添加shift_mean模块
        """
        self.shift_mean = args.shift_mean
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)
        self.la = LAM_Module(n_feats)
        """
        n_feats*11 : n_rgs=10 + rgs之后的一个卷积层
        与论文所展示结构略有区别：将最后卷积层的输出，也输入到lam中
        """
        self.last_conv = nn.Conv2d(n_feats*11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats*2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        if self.shift_mean:
            x = self.sub_mean(x)

        x = self.head(x)
        res = x
        #pdb.set_trace()
        """
        属性_modules以OrderDict的形式，返回所有子模块
        字典方法items()，返回包含字典中所有（key, value）元组的列表
        """
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            #print(name)
            """
            unsqueeze在指定位置，增加一个大小为1的维度，创建源tensor的视图，输入张量与输出张量共享内存，改变输出张量，输入张量也会变化
            unsqueeze(1) 中的”1“指的是第二个位置
            batch, channel, height, width, depth
            新增加的维度1，代表num_midlayer
            占据了原先的channel通道，在卷积中，当作channel处理
            
            存在问题：新增加了一个维度，与其他处理结果形状不同
            """
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                """
                在维度1上，拼接两个张量
                """
                res1 = torch.cat([res.unsqueeze(1),res1],1)
        """
        out1:rgs的输出
        out2:lam的输出
        经过self.last_conv的处理：输入11通道，输出1通道，整合各层次之间的信息
        ？？？疑惑：经过self.last_conv整合后，第二个维度，变成1，但依旧不是之前的通道维度（此时在第三个维度），与其他信息存在差异
        """
        #res = self.body(x)
        out1 = res
        #res3 = res.unsqueeze(1)
        #res = torch.cat([res1,res3],1)
        res = self.la(res1)
        out2 = self.last_conv(res)

        """
        out1：经过rgs和一个卷积层得到输出，将输出输入到csam中，得到out1
        ？？？疑惑：lam与csam的输出在第二个维度上的差异（尝试查看csam代码）
        """
        out1 = self.csa(out1)
        out = torch.cat([out1, out2], 1)
        """
        ？？？self.last
        
        """
        res = self.last(out)
        """
        res：long skip、lam和csam的整合结果
        """
        res += x
        #res = self.csa(res)

        x = self.tail(res)

        if self.shift_mean:
            x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))