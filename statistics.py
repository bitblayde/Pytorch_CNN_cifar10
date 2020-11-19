import torch
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt
import os

dataset_path = "./dataset/"

if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)


X_train = datasets.CIFAR10(dataset_path, train=True, transform = transforms.Compose([ transforms.ToTensor() ]) )
X_validation = datasets.CIFAR10(dataset_path, train=False, transform = transforms.Compose([ transforms.ToTensor() ]))

imgs, labels  = X_train[100]

imgs = torch.stack([img for img, labels in X_train ], dim = 3)

imgs = imgs.view(3, -1)

dir_stats = './utilities'
file = 'normalize.pt'
path_complete = os.path.join(dir_stats, file)

if not os.path.isdir(dir_stats):
    os.mkdir(dir_stats)

torch.save([imgs.mean(dim=1), imgs.std(dim=1)], path_complete)
