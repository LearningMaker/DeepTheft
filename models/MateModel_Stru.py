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


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lstm_pointwise = nn.Sequential(
            nn.Conv1d(64, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_channels*3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.block = Block(in_channels, out_channels)

    def forward(self, x, copy_x, lstm_x):
        up_x = F.interpolate(x, size=copy_x.size(2), mode="linear", align_corners=True)
        lstm_x = F.interpolate(lstm_x, size=copy_x.size(2), mode="linear", align_corners=True)
        lstm_x = self.lstm_pointwise(lstm_x)

        out = torch.cat([up_x, copy_x, lstm_x], dim=1)
        out = self.pointwise(out)
        out = self.block(out)
        return out


class MiddleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = Block(in_channels, out_channels)
        self.pointwise1 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm1d(in_channels),
        )
        self.lstm = nn.LSTM(out_channels, out_channels, bidirectional=True)
        self.pointwise2 = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, 1, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        conv_x = self.block(x)

        lstm_x = self.pointwise1(conv_x)
        lstm_x, _ = self.lstm(lstm_x.permute(2, 0, 1))
        lstm_x = self.pointwise2(lstm_x.permute(1, 2, 0))

        return lstm_x, conv_x


class FinalBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv1d(in_channels, num_classes, 1, 1),
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

        self.middle = MiddleBlock(filter[3], filter[3])

        self.up_conv4 = UpBlock(filter[3], filter[2])
        self.up_conv3 = UpBlock(filter[2], filter[1])
        self.up_conv2 = UpBlock(filter[1], filter[0])
        self.up_conv1 = UpBlock(filter[0], filter[0])

        self.final = FinalBlock(filter[0], num_classes)

    def forward(self, x):
        x1, down_x1 = self.down_conv1(x)
        x2, down_x2 = self.down_conv2(down_x1)
        x3, down_x3 = self.down_conv3(down_x2)
        x4, down_x4 = self.down_conv4(down_x3)

        lstm_x, conv_x = self.middle(down_x4)

        up_x4 = self.up_conv4(conv_x, x4, lstm_x)
        up_x3 = self.up_conv3(up_x4, x3, lstm_x)
        up_x2 = self.up_conv2(up_x3, x2, lstm_x)
        up_x1 = self.up_conv1(up_x2, x1, lstm_x)

        out = self.final(up_x1)
        return out