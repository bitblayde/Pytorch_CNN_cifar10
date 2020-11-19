import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot


class Basic_block(nn.Module):
    offset = 0

    def __init__(self, input_filters, output_filters, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_filters, output_filters, 3, stride, 1, bias = False),
            nn.BatchNorm2d(output_filters),
            nn.ReLU(),
            nn.Conv2d(output_filters, output_filters, 3, 1, 1, bias = False),
            nn.BatchNorm2d(output_filters)
        )
        self.shortcut = nn.Sequential()

        if input_filters != (output_filters << self.offset) or stride != 1:
            self.shortcut = self.identity_block(input_filters, output_filters << self.offset, stride)

    def identity_block(self, input_filters, output_filters, stride):
        return nn.Sequential(
            nn.Conv2d(input_filters, output_filters, 1, stride, bias = False),
            nn.BatchNorm2d(output_filters)
        )

    def forward(self, x):
        out = self.block(x)
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck_block(nn.Module):
    offset = 2

    def __init__(self, filters_in, filters_out, stride):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(filters_in, filters_out, kernel_size = 1, bias = False),
            nn.BatchNorm2d(filters_out),
            nn.ReLU(),
            nn.Conv2d(filters_out, filters_out, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(filters_out),
            nn.ReLU(),
            nn.Conv2d( filters_out, (filters_out << self.offset), kernel_size = 1, bias = False),
            nn.BatchNorm2d(filters_out << self.offset),
        )
        self.shortcut = nn.Sequential()

        if filters_in != (filters_out << self.offset) or stride != 1:
            self.shortcut = self.__identity_block__(filters_in, filters_out, stride)


    def __identity_block__(self, filters_in, filters_out, stride):
        return nn.Sequential(
            nn.Conv2d(filters_in, filters_out << self.offset, 1, stride, bias = False),
            nn.BatchNorm2d(filters_out << self.offset)
        )

    def forward(self, x):
        output = self.block(x)
        output += self.shortcut(x)
        return F.relu(output)


class Resnet(nn.Module):
    def __init__(self, stage, block, classes, init_weights = False):
        super().__init__()
        self.init_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.filters = [64, 128, 256, 512]
        self.input = self.filters[0]
        self.blocks = self.__make_residual__(stage, block, 64)
        self.fc = nn.Linear( 512 << block.offset, classes )

        if init_weights:
            self.__init_weights__()

    def __init_weights__(self):
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(i, nn.BatchNorm2d):
                nn.init.constant_(i.weight, 0.5)
                nn.init.zeros_(i.bias)

    def __make_residual__(self, stage, block, input_filters, stride = None):

        arch = []
        for layer in range(4):
            if layer == 0:
                stride = [1] + [1]*(stage[layer] - 1)
            else:
                stride = [2] + [1]*(stage[layer] - 1)

            for i in stride:
                arch.append( block( self.input, self.filters[layer], i ) )
                self.input = self.filters[layer] << block.offset
        return nn.Sequential(*arch)

    def forward(self, x):
        x = self.init_block(x)
        x = self.blocks(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

resnet_architecture = {
"18":[[2, 2, 2, 2], Basic_block],
"34":[[3, 4, 6, 3], Basic_block],
"50":[[3, 4, 6, 3], Bottleneck_block],
"101":[[3, 4, 23, 3], Bottleneck_block],
"152":[[3, 8, 36, 3], Bottleneck_block]
}

def resnet_model(model, classes):
    return Resnet(resnet_architecture[model][0], resnet_architecture[model][1], classes, True)

def testing(option = "101"):
    model = resnet_model(option, 10)
    y = model(torch.randn(6, 3, 32, 32))
    make_dot(y, params = dict(list(model.named_parameters() ))).render("Resnet" + option, format="png")

#testing()
