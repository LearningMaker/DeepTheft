import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import h5py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

models = {'vgg':0, 'vgg_bn':1, 'resnet_basicblock':2, 'resnet_bottleneck':3, 'custom_net':4, 'custom_net_bn':5}
samples = [256, 256, 2592, 2592, 2728, 2728]
train_samples = [12, 12, 116, 116, 122, 122]
test_samples = [26, 26, 268, 268, 282, 282]
test_index = []
np.random.seed(0)
for i in range(len(samples)):
    index = np.random.choice(samples[i], test_samples[i], replace=False)
    test_index.append(index)
train_index = []
np.random.seed(1)
for i in range(len(samples)):
    index = np.random.choice(samples[i], train_samples[i], replace=False)
    train_index.append(index)


class Normalization(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        _range = np.max(input, axis=0) - np.min(input, axis=0) + 1e-7
        input = (input - np.min(input, axis=0)) / _range
        return input


class Resize(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def forward(self, inputs):
        out = [inputs[int(i * inputs.shape[0] / self.length)] for i in range(self.length)]
        out = np.array(out).transpose([1, 0])
        return out


class ToTargets(torch.nn.Module):
    def __init__(self, mode, label):
        super().__init__()
        self.mode = mode
        self.label = label

    def forward(self, targets):
        if self.mode == 'kernel_size':
            targets = targets[self.label]
            targets = (targets - 1) / 2
            targets = targets - 1 if targets == 3 else targets
        if self.mode == 'stride':
            targets = targets[self.label]
            targets = targets - 1
        if self.mode == 'out_channels':
            targets = targets[self.label]
            targets = np.log2(targets) - 6
        return targets


class Rapl(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        (self.feature, self.label) = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        feat, lab = self.feature[index][:, 1:3], self.label[index]
        feat = self.transform(feat) if self.transform is not None else feat
        lab = self.target_transform(lab) if self.target_transform is not None else lab
        return feat, lab

    def __len__(self):
        return len(self.feature)


class RaplLoader(object):
    def __init__(self, batch_size, mode, num_workers=0):
        self.label = {'in_channels': 0, 'out_channels': 1, 'kernel_size': 2,
                      'stride': 3, 'padding': 4, 'dilation': 5,
                      'groups': 6, 'input_size': 7, 'output_size': 8}[mode]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = {'out_channels': 6, 'kernel_size': 3, 'stride': 2}[mode]
        self.train, self.val = self.preprocess()

        self.transform = transforms.Compose([
            Normalization(),
            Resize(1024),
        ])
        self.target_transform = transforms.Compose([
            ToTargets(mode, self.label),
        ])

    def preprocess(self):
        train_x, train_y = [], []
        val_x, val_y = [], []
        x = h5py.File(r'../datasets/data.h5', 'r')
        y = h5py.File(r'../datasets/hp.h5', 'r')

        for k in x['data'].keys():
            if k.split(')')[1] == '224':
                d = x['data'][k][:]
                pos = x['position'][k][:]
                hp = y[k][:]
                hp = hp[hp[:, 0] != -1]
                hp = hp[hp[:, -1] != -1]

                test_indexes = test_index[models[k.split(')')[0]]]
                n = int(k.split(')')[-1])
                hp_index = 0
                for (i, j) in pos:
                    if d[:, -1][i] == 0:
                        if n in test_indexes:
                            val_x.append(d[i:j + 1, :-1])
                            val_y.append(hp[hp_index])
                        else:
                            train_x.append(d[i:j + 1, :-1])
                            train_y.append(hp[hp_index])
                        hp_index += 1
                assert hp_index == len(hp)

        return (train_x, train_y), (val_x, val_y)

    def loader(self, data, shuffle=False, transform=None, target_transform=None):
        dataset = Rapl(data, transform=transform, target_transform=target_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return dataloader

    def get_loader(self):
        trainloader = self.loader(self.train, shuffle=True, transform=self.transform, target_transform=self.target_transform)
        valloader = self.loader(self.val, transform=self.transform, target_transform=self.target_transform)
        return trainloader, valloader
