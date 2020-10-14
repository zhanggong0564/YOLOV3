import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.models.resnet
import math



class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x*torch.tanh(F.softplus(x))
class BasicConv(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride =1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = Mish()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Resblock(nn.Module):
    def __init__(self,channels,hidden_channels=None):
        super(Resblock, self).__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.block = nn.Sequential(
            BasicConv(channels,hidden_channels,1),
            BasicConv(hidden_channels,channels,3)
        )
    def forward(self, x):
        out = self.block(x)
        out = x+out
        return out



class Resblock_body(nn.Module):
    def __init__(self,in_ch,out_ch,num_block,first):
        super(Resblock_body, self).__init__()
        self.downsample_conv = BasicConv(in_ch,out_ch,3,stride=2)
        if first:
            self.split_conv0 = BasicConv(out_ch,out_ch,1)
            self.split_conv1 = BasicConv(out_ch,out_ch,1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_ch,hidden_channels=out_ch//2),
                BasicConv(out_ch,out_ch,1)
            )
            self.concat = BasicConv(out_ch*2,out_ch,1)
        else:
            self.split_conv0 = BasicConv(out_ch,out_ch//2,1)
            self.split_conv1 = BasicConv(out_ch,out_ch//2,1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_ch//2) for _ in range(num_block)],
                BasicConv(out_ch//2,out_ch//2,1)
            )
            self.concat = BasicConv(out_ch,out_ch,1)
    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.cat([x1,x0],dim=1)
        x = self.concat(x)
        return x
class CSPDarknet(nn.Module):
    def __init__(self,layers):
        super(CSPDarknet, self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3,self.inplanes,kernel_size = 3,stride = 1)
        self.feature_channels = [32,64,128,256,512]
        self.stages = nn.ModuleList([
            Resblock_body(self.inplanes,self.feature_channels[0],layers[0],first = True),
            Resblock_body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            Resblock_body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            Resblock_body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            Resblock_body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False),
        ])

        # for m in self.modules():
        #     if isinstance(m,nn.Conv2d):
        #         n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
        #         m.weight.data.normal_(0,math.sqrt(2./n))
        #     elif isinstance(m,nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x =self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x1 = self.stages[2](x)
        x2 = self.stages[3](x1)
        x3 = self.stages[4](x2)
        return x1,x2,x3


def darknet53(pretraind = False,**kwargs):
    model  = CSPDarknet([1,2,8,8,4])
    if pretraind:
        if isinstance(pretraind,str):
            model.load_state_dict(torch.load(pretraind))
        else:
            raise  Exception("darknet request a pretrained path. got [{}]".format(pretraind))
    return model


if __name__ == '__main__':
    x = torch.randn(1,3,416,416)
    model = darknet53()
    x1, x2, x3 = model(x)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)