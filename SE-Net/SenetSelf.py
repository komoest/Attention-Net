import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, block_channel, stride, r=16, first=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, block_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(block_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(block_channel, block_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(block_channel),
            nn.ReLU()
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(block_channel, block_channel // r, kernel_size=1, stride=1, bias=False),  # 1*1conv相当于全连接，r选取16
            nn.ReLU(),
            nn.Conv2d(block_channel // r, block_channel, kernel_size=1, stride=1, bias=False),  # 1*1conv相当于全连接，r选取16
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, block_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(block_channel),
        )
        self.relu = nn.ReLU(inplace=True)
        self.first = first

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        se = self.se(out)
        out = out * se
        if self.first:
            shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channel, channels, stride, r=16, first=False):
        super(Bottleneck, self).__init__()
        channel1, channel2, channel3 = channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, channel1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channel1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel1, channel2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel2, channel3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel3),
            nn.ReLU()
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channel3, channel3 // r, kernel_size=1, stride=1, bias=False),  # 1*1conv相当于全连接，r选取16
            nn.ReLU(),
            nn.Conv2d(channel3 // r, channel3, kernel_size=1, stride=1, bias=False),  # 1*1conv相当于全连接，r选取16
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, channel3, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channel3),
        )
        self.first = first

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        se = self.se(out)
        out = out * se
        if self.first:
            shortcut = self.shortcut(x)
        out = out + shortcut
        out = self.relu(out)

        return out


class FullNet(nn.Module):
    def __init__(self, block, in_channel, channels, block_num, r=16, classes=10):
        super(FullNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self.makelayer(block, in_channel[0], channels[0], block_num[0], 1, r)
        self.conv3 = self.makelayer(block, in_channel[1], channels[1], block_num[1], 2, r)
        self.conv4 = self.makelayer(block, in_channel[2], channels[2], block_num[2], 2, r)
        self.conv5 = self.makelayer(block, in_channel[3], channels[3], block_num[3], 2, r)
        self.glbp = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3], classes)

    def makelayer(self, block, in_channel, channels, block_num, stride, r):
        layers = []
        layers.append(block(in_channel, channels, stride=stride, r=r, first=True))
        for i in range(1, block_num):
            layers.append(block(channels, channels, stride=1, r=r))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.glbp(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def SeNet18():
    in_channel = [64, 64, 128, 256]
    channels = [64, 128, 256, 512]
    block_num = [2, 2, 2, 2]
    model = FullNet(BasicBlock, in_channel, channels, block_num)
    return model


def SeNet34():
    in_channel = [64, 64, 128, 256]
    channels = [64, 128, 256, 512]
    block_num = [3, 4, 6, 3]
    model = FullNet(BasicBlock, in_channel, channels, block_num)
    return model


def SeNet50():
    in_channel = [64, 256, 512, 1024]
    channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    block_num = [3, 4, 6, 3]
    model = FullNet(Bottleneck, in_channel, channels, block_num)
    return model


def SeNet101():
    in_channel = [64, 256, 512, 1024]
    channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    block_num = [3, 4, 23, 3]
    model = FullNet(Bottleneck, in_channel, channels, block_num)
    return model


def SeNet152():
    in_channel = [64, 256, 512, 1024]
    channels = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
    block_num = [3, 8, 36, 3]
    model = FullNet(Bottleneck, in_channel, channels, block_num)
    return model
