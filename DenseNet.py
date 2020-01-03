import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    def __init__(self,nChannels,growthRate):
        super(Bottleneck,self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels,interChannels,kernel_size=1,
                               stride=1,bias=False)
        self.bn2 = nn.BatchNorm3d(interChannels)
        self.conv2 = nn.Conv3d(interChannels,growthRate,kernel_size=3,
                               stride=1,padding=1,bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #先进行BN（pytorch的BN已经包含了Scale）,然后进行relu,conv1起到bottleneck的作用
        out = self.conv1(F.relu(self.bn1(x)))
        #out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        #out = self.dropout(out)
        out = torch.cat((x,out),1)
        return out

class Transitions(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transitions,self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=1,bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(F.leaky_relu(self.bn1(x)))
        #out = self.dropout(out)
        out = F.avg_pool3d(out,2)
        return out

class SingleLayer(nn.Module):
    def __init__(self,nChannels,growthRate):
        super(SingleLayer,self).__init__()
        self.bn1 = nn.BatchNorm3d(nChannels)
        self.conv1 = nn.Conv3d(nChannels,growthRate,kernel_size=3,
                               padding=1,bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat(x,out)
        return out


class DenseNet(nn.Module):
    def __init__(self,growthRate,depth,reduction,nClasses,bottleneck,verbose=False):
        super(DenseNet,self).__init__()
        #DenseBlock中非线性变换模块的个数
        nNoneLinears = (depth-4)//3
        if bottleneck:
            nNoneLinears //=2#3

        self.verbose = verbose

        nChannels = growthRate#64
        self.conv1 = nn.Conv3d(1,nChannels,kernel_size=3,padding=1,bias=False)#1,64
        self.denseblock1 = self._make_dense(nChannels,growthRate,nNoneLinears,bottleneck)#64,32
        nChannels += nNoneLinears*growthRate#3*32
        nOutChannels = int(math.floor(nChannels*reduction))        #向下取整
        self.transition1 = Transitions(nChannels,nOutChannels)

        nChannels = nOutChannels
        self.denseblock2 = self._make_dense(nChannels,growthRate,nNoneLinears,bottleneck)
        nChannels += nNoneLinears*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.transition2 = Transitions(nChannels,nOutChannels)

        nChannels = nOutChannels
        self.denseblock3 = self._make_dense(nChannels, growthRate, nNoneLinears, bottleneck)
        nChannels += nNoneLinears * growthRate

        self.bn1 = nn.BatchNorm3d(nChannels)
        self.fc = nn.Linear(nChannels,nClasses)

        self.sigmoid = nn.Sigmoid()

        self.LeakyRelu = nn.LeakyReLU()

        #参数初始化
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self,nChannels,growthRate,nDenseBlocks,bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels,growthRate))
            else:
                layers.append(SingleLayer(nChannels,growthRate))
            nChannels+=growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.verbose:
            print('conv1 output: {}'.format(x.shape))
        x = self.transition1(self.denseblock1(x))
        if self.verbose:
            print('transition1 output: {}'.format(x.shape))
        x = self.transition2(self.denseblock2(x))
        if self.verbose:
            print('transition2 output: {}'.format(x.shape))
        x = self.denseblock3(x)
        if self.verbose:
            print('denseblock3 output: {}'.format(x.shape))
        x = torch.squeeze(F.avg_pool3d(F.relu(self.bn1(x)),8))
        if self.verbose:
            print('squeeze output: {}'.format(x.shape))
        x = self.sigmoid(self.fc(x))
        # x = self.LeakyRelu(self.fc(x))
        if self.verbose:
            print('log_softmax output: {}'.format(x.shape))
        return x
