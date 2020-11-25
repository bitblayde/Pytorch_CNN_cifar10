import torch
import torch.nn as nn

class transition_block(nn.Module):
    def __init__(self, input_filters, output_filters):
        super().__init__()
        self.transition_core = nn.Sequential(
            nn.BatchNorm2d(input_filters),
            nn.ReLU(),
            nn.Conv2d( input_filters, output_filters, kernel_size = 1, bias = False ),
            nn.AvgPool2d(kernel_size = 2, stride = 2)
        )

    def forward(self, x):
        return self.transition_core(x)


class botleneck_block(nn.Module):
    def __init__(self, input_filters, output_filters):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(input_filters),
            nn.ReLU(),
            nn.Conv2d( input_filters, output_filters, kernel_size = 1, bias = False )
        )


        self.block2 = nn.Sequential(
            nn.BatchNorm2d(output_filters),
            nn.ReLU(),
            nn.Conv2d( output_filters, output_filters >> 2, kernel_size = 3, padding = 1, bias = False )
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)

        return torch.cat([out, x], 1)


class DenseNet_core(nn.Module):
    def __init__(self, block, stage, in_channels = 3, k = 12, theta = 0.5):
        super().__init__()
        self.k = k
        current_filters = self.k << 1
        self.conv1 = nn.Conv2d(in_channels, current_filters, kernel_size = 3, padding = 1, bias = False)

        self.dense_block1 = self.dense_block(block, stage[0], current_filters)
        current_filters += stage[0] * self.k
        self.transition1 = transition_block(input_filters = current_filters, output_filters = int(current_filters*theta))
        current_filters = int(current_filters*theta)

        self.dense_block2 = self.dense_block(block, stage[1], current_filters)
        current_filters += stage[1] * self.k
        self.transition2 = transition_block(input_filters = current_filters, output_filters = int(current_filters*theta))
        current_filters = int(current_filters*theta)

        self.dense_block3 = self.dense_block(block, stage[2], current_filters)
        current_filters += stage[2] * self.k
        self.transition3 = transition_block(input_filters = current_filters, output_filters = int(current_filters*theta))
        current_filters = int(current_filters*theta)

        self.dense_block4 = self.dense_block(block, stage[3], current_filters)
        current_filters += stage[3] * self.k

        self.bn = nn.BatchNorm2d(current_filters)
        self.relu = nn.ReLU(True)
        self.avg = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(current_filters, 10)

    def dense_block(self, block, stage, initial_filters):
        layers = []

        for i in range(stage):
            layers.append( block( input_filters = initial_filters, output_filters = self.k << 2 ) )
            initial_filters += self.k

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.transition2( self.dense_block2(x) )
        x = self.transition3( self.dense_block3(x) )
        x = self.dense_block4(x)

        x = self.avg(self.bn(self.relu(x)))

        x = x.view(x.size(0), -1)

        return self.fc1(x)


class DenseNet():
    def __init__(self, custom = False, stages = None, block = None, k = 12, theta = 0.5, img_channels = 3):
        if custom:
            self.stages = stages
            self.block = block

        self.k = k
        self.theta = theta
        self.img_channels = img_channels

    def dense_121(self):
        return DenseNet_core(botleneck_block, [6, 12, 24, 16], in_channels = self.img_channels, k = self.k, theta = self.theta)
    def dense_169(self):
        return DenseNet_core(botleneck_block, [6, 12, 32, 32], in_channels = self.img_channels, k = self.k, theta = self.theta)
    def dense_201(self):
        return DenseNet_core(botleneck_block, [6, 12, 48, 32], in_channels = self.img_channels, k = self.k, theta = self.theta)
    def dense_264(self):
        return DenseNet_core(botleneck_block, [6, 12, 64, 48], in_channels = self.img_channels, k = self.k, theta = self.theta)

    def __call__(self):
        return DenseNet_core(self.block, self.stages, self.img_channels, self.k, self.theta)

def custom_densenet():
    model = DenseNet(True, [6, 12, 24, 16], botleneck_block, 24, 0.5, 3)()
    x = torch.randn(1,3,32,32)
    y = model(x)
    print(y)

def test():
    core_densenet = DenseNet(False)
    model = core_densenet.dense_121()
    x = torch.randn(1,3,32,32)
    y = model(x)
    print(y)

custom_densenet()
test()
