import torch
from torchvision import datasets
from torchvision import transforms
import torch.optim
from matplotlib import pyplot as plt
from torchviz import make_dot
import models.simple_network as network
import torch.nn as nn
import os
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

import numpy as np

dataset_path = "./dataset/"
cm_path = "./utils"
normalize_data_path = "./utilities/normalize.pt"


parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('-v', '--vbs', default = "verbose", help='verbose')
parser.add_argument('-e', '--epochs', default = 40, help='training epochs')
parser.add_argument('-b', '--batch', default = 512, help='batch size')
parser.add_argument('-l', '--lr', default = 1e-1, help='learning rate')

args = parser.parse_args()

params = {"verbose" : args.vbs, "epochs" : int(args.epochs), "batches" : int(args.batch), "learning rate" : float(args.lr)}

class_names = ['airplane','automobile','bird','cat','deer', 'dog','frog','horse','ship','truck']

normalization_data = torch.load(normalize_data_path, map_location=torch.device('cpu'))

transformations_train = transforms.Compose([
    transforms.RandomHorizontalFlip(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((normalization_data[0]), (normalization_data[1]))
])

transformations_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((normalization_data[0]), (normalization_data[1]))
])

X_train = datasets.CIFAR10(dataset_path, train=True, transform = transformations_train)
X_validation = datasets.CIFAR10(dataset_path, train=False, transform = transformations_test)

train_loader = torch.utils.data.DataLoader(X_train, batch_size = params["batches"], shuffle = True)
test_loader = torch.utils.data.DataLoader(X_validation, batch_size = params["batches"], shuffle = False)

model = network.Network(3)
device = ( torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') )

if params["verbose"].upper() == "VERBOSE":
    print("\n", model)
    print("\t\t\t\tParams\n", params)
    print("\ndevice \t\t", device)
    x = torch.zeros(1, 3, 32, 32, dtype=torch.float, requires_grad=False)
    x = model(x)
    make_dot(x, params = dict(list(model.named_parameters() ))).render(os.path.join(cm_path ,"network_architecture"), format="png")

model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = params["learning rate"], nesterov = True, momentum = 0.9, weight_decay = 1e-3)
loss = nn.CrossEntropyLoss()

train_loss_hist = np.zeros(params["epochs"])
train_acc_hist = np.zeros(params["epochs"])

model.train()
for EPOCH in range(1, params["epochs"] + 1):
    loss_train = 0
    total = 0
    correctas = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        current_loss = loss(out, labels)

        _, pred = torch.max(out, dim=1)
        total += labels.shape[0]
        correctas += int((pred == labels).sum())

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        loss_train += current_loss.item()

    train_loss_hist[EPOCH-1] = loss_train/params["batches"]
    train_acc_hist[EPOCH-1] = correctas/total

    print("Acc {:.4f} \tloss {:.4f}, \t{:d}/{:d}".format( (correctas/total), (loss_train/params["batches"]), EPOCH, params["epochs"] ))


model.eval()
for name, loader_iter in [('train', train_loader), ('val', test_loader)]:
    total = 0
    correctas = 0
    predicted = []
    real = []
    with torch.no_grad():
        for imgs, labels in loader_iter:
            true_labels_cpu = labels.numpy()
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            _, pred = torch.max(out, dim=1)
            total += labels.shape[0]
            correctas += int((pred == labels).sum())

            pred_labels_cpu = pred.cpu().numpy()

            predicted = np.concatenate([predicted, pred_labels_cpu ])
            real = np.concatenate([real, true_labels_cpu ])

    cm = confusion_matrix( predicted, real, labels=torch.arange(0, 10))
    df_cm = pd.DataFrame(cm, index = [i for i in class_names], columns = [i for i in class_names])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10},fmt="d")
    plt.title(str(float(correctas/total)))
    plt.savefig( os.path.join(cm_path, name) + ".png" )
    plt.show()
    print("Particion {}, {:.4f}".format(name, correctas/total))

plt.title("Training. Loss and Accuracy")
plt.grid(True)
plt.plot(np.arange(params["epochs"]), train_loss_hist, label = "loss")
plt.plot(np.arange(params["epochs"]), train_acc_hist, label = "accuracy")
plt.legend()
plt.savefig( os.path.join(cm_path, "hist.png") )
plt.show()
