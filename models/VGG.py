import torch
import torch.nn as nn

import vgg_configure_options as vgg_configure_options

class VGG(nn.Module):
    def __init__(self, input_dim, scheme):
        super().__init__()
        self.option = scheme
        last_convolutional_filters, self.convolutional_core = self.build_vgg_blocks(input_dim)
        self.sequential_core = self.fc_layers( last_convolutional_filters )

    def forward(self, x):
        x = self.convolutional_core(x)
        x = x.view(x.size(0), -1)
        return self.sequential_core(x)


    def build_vgg_blocks(self, input_dim):
        last_channels = input_dim
        convolutional = []

        for param in self.option:
            if param != 'P':
                convolutional.append( nn.Conv2d( in_channels = last_channels, out_channels = param, kernel_size = (3, 3), padding = 1, stride = 1 ) )
                convolutional.append( nn.ReLU() )
                convolutional.append( nn.BatchNorm2d( param ) )
                last_channels = param
            else:
                convolutional.append( nn.MaxPool2d( 2, 2 ) )

        convolutional.append( nn.AdaptiveMaxPool2d( (2, 2) ) )

        return last_channels, nn.Sequential(*convolutional)

    def fc_layers(self, input_dim):
        fc_module = []

        fc_module.append( nn.Linear(input_dim*2*2, 4096) )
        fc_module.append( nn.ReLU() )

        fc_module.append( nn.Linear(4096, 4096) )
        fc_module.append( nn.ReLU() )

        fc_module.append( nn.Linear(4096, 1000) )
        fc_module.append( nn.ReLU() )

        fc_module.append( nn.Linear(1000, 10) )
        return nn.Sequential(*fc_module)

def testing():
	x = torch.randn(size = (5, 3, 32, 32))
	network = VGG(x.shape[1], vgg_configure_options.vgg_options["B"])
	output = network(x)
	print(output)

#testing()
