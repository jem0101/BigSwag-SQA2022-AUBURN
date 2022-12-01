import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(UpSample, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(in_planes, out_planes, 4, 2, 1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.upsample(self.relu(self.bn1(x)))
        if self.droprate > 0.0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.image_size = kwargs["input_size"]
        self.z_dim = kwargs["z_dim"]
        self.n_channel = kwargs["n_channel"]

        self.feature_dim = [
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64)]
        self.dense_layer = [3, 3, 3, 3]

        growth_rate = 3
        reduction = 0.25
        bottleneck = False
        dropRate = 0.0
        noise_rate = 2

        in_planes = growth_rate
        if bottleneck == True:
            self.dense_layer = [i // 2 for i in self.dense_layer]
            block = BottleneckBlock
        else:
            block = BasicBlock

        self.linear = nn.Linear(self.z_dim, int(3 * np.prod(self.feature_dim[0])))

        # # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        list_module = []
        for sb, dl in zip(self.feature_dim, self.dense_layer):
            if sb[0] == self.image_size: break
            sub_module = []
            in_planes = int(in_planes * noise_rate)
            sub_module.append(DenseBlock(dl, in_planes, growth_rate, block, dropRate))
            in_planes = int(in_planes + dl * growth_rate)
            sub_module.append(UpSample(in_planes, int(in_planes * reduction)))
            in_planes = int(in_planes * reduction)
            list_module.append(nn.Sequential(*sub_module))

        self.module_list = nn.ModuleList(list_module)
        self.module_out = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, self.n_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, z_input):
        # make z_input
        output = self.linear(z_input)
        output = output.view(z_input.shape[0], 3, *self.feature_dim[0])
        output = self.conv1(output)

        for f_dim, sub_module in zip(self.feature_dim, self.module_list):
            noise = torch.rand(output.shape).type(output.type()).normal_(-1, 1)
            output = torch.cat((output, noise), 1)
            assert f_dim[0] == output.shape[-1], "shape not same, expect {} get {}".format(f_dim, output.shape[-2:])
            output = sub_module(output)

        return self.module_out(output)


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.image_size = kwargs["input_size"]
        self.z_dim = kwargs["z_dim"]
        self.n_channel = kwargs["n_channel"]
        self.ndf = kwargs["ndf"]
        self.output_dimension = kwargs["output_disc"]

        self.dense_layer = [3, 3, 3]

        growth_rate = 3
        reduction = 0.5
        bottleneck = False
        dropRate = 0.0

        in_planes = 2 * growth_rate
        if bottleneck == True:
            self.dense_layer = [i // 2 for i in self.dense_layer]
            block = BottleneckBlock
        else:
            block = BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(self.dense_layer[0], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + self.dense_layer[0] * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(self.dense_layer[1], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + self.dense_layer[1] * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(self.dense_layer[2], in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + self.dense_layer[2] * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, self.output_dimension)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, input_data):
        if self.n_channel == 1:
            input_data = torch.cat((input_data, input_data, input_data), 1)
        out = self.conv1(input_data)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)
