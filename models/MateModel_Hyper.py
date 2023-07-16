import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels=None):
        super().__init__()
        inner_channels = out_channels // 2 if inner_channels is None else inner_channels

        self.conv1 = nn.Conv1d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(inner_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = Block(in_channels, out_channels)
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        block_x = self.block(x)
        down_x = self.pool(block_x)
        return block_x, down_x


class FinalBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x):
        out = self.classifier(x)
        return out


class Model(nn.Module):
    def __init__(self, num_classes, input_channels=2):
        super().__init__()
        n = 8
        filter = [n, n * 2, n * 4, n * 8]

        self.down_conv1 = DownBlock(input_channels, filter[0])
        self.down_conv2 = DownBlock(filter[0], filter[1])
        self.down_conv3 = DownBlock(filter[1], filter[2])
        self.down_conv4 = DownBlock(filter[2], filter[3])

        self.final = FinalBlock(filter[3], num_classes)

    def forward(self, x):
        _, down_x1 = self.down_conv1(x)
        _, down_x2 = self.down_conv2(down_x1)
        _, down_x3 = self.down_conv3(down_x2)
        _, down_x4 = self.down_conv4(down_x3)

        out = self.final(down_x4)
        return out

