import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = input_dim, out_channels = 16, kernel_size = (3, 3), padding = 1)
        self.batchnormalization1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 64, kernel_size = (3, 3), padding = 1)
        self.batchnormalization2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = (3, 3), padding = 1)
        self.batchnormalization3 = nn.BatchNorm2d(num_features=32)

        self.fc1 = nn.Linear(in_features = 32*4*4, out_features = 1000)
        self.fc2 = nn.Linear(in_features = 1000, out_features = 10)

    def forward(self, x):
        x = F.max_pool2d( torch.nn.functional.relu( self.conv1(x) ), 2 )
        x = self.batchnormalization1(x)
        x = F.max_pool2d( torch.nn.functional.relu( self.conv2(x) ), 2 )
        x = self.batchnormalization2(x)
        x = F.max_pool2d( torch.nn.functional.relu( self.conv3(x) ), 2 )
        x = self.batchnormalization3(x)

        x = torch.nn.functional.adaptive_max_pool2d(x, (4, 4))

        x = x.view(x.size(0), -1)

        x = torch.nn.functional.relu(self.fc1(x))

        return self.fc2(x)
