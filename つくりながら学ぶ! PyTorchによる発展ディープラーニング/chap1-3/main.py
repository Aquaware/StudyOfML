import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

from DataSet import DataSet
from ImageTransform import ImageTransform





# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


def fileList(dir_path, extension=None):
    out = []
    for holder, subholders, files in os.walk(dir_path):
        for file in files:
            if extension is not None:
                if file.find(extension) >= 0:        
                    out.append(holder + '/' + file)
            else:
                out.append(holder + '/' + file)
    return out

def train(net, dataloaders, criterion, optimizer, epoch_nums):
    for i in range(epoch_nums):
        print('Epoch : ', i)
        for phase in ['train', 'test']:
            if phase == 'train':
                net.train()
            elif phase == 'test':
                net.eval()
            loss_sum = 0.0
            correct_count = 0
            if i == 0 and phase == 'train':
                continue

            for inputs, labels in tqdm(dataloaders[phase]):
                optimizer.zero_grad()
    return

def torchImage2PIL(image_data):
    image = image_data.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    return image

def createModel():
    net = models.vgg16(pretrained=True)
    #print(net)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()
    #print(net)
    return net

def createDataLoader():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32
    
    train_list = fileList('../data/train/', extension='jpg')
    test_list = fileList('../data/val/', extension='jpg')

    train_dataset = DataSet('train', ImageTransform(size, mean, std), train_list)
    test_dataset = DataSet('test', ImageTransform(size, mean, std), test_list)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return (train_dataloader, test_dataloader)


# ------
    
def main():
    train_dataloader, test_dataloader = createDataLoader()
    iterator = iter(train_dataloader)
    inputs, labels = next(iterator)
    print(inputs.size())
    print(labels)

    net = createModel()
    net.train()
    print('Neural Networking done.')

    loss = nn.CrossEntropyLoss()

    modify_parameters = ['classifier.6.weight', 'classifier.6.bias']
    parameters = []
    for name, param in net.named_parameters():
        if name in modify_parameters:
            param.requires_grad = True
            parameters.append(param)
            print(name)
        else:
            param.requires_grad = False
    print(parameters)
    optimizer = optim.SGD(params=parameters, lr=0.001, momentum=0.9)
    
    
def test1():
    filepath = './data/goldenretriever-3724972_640.jpg'
    image = Image.open(filepath)
    plt.imshow(image)
    plt.show()
    transform = ImageTransform(size, mean, std)
    image1 = transform(image, phase='train')
    plt.imshow(torchImage2PIL(image1))
    plt.show()
    return

def test2():
    dir_path = '../data/train/'
    train_data = DataSet('train', size, mean, std, dir_path)

    dir_path = '../data/val/'
    test_data = DataSet('test', size, mean, std, dir_path)

    index = 0
    print(train_data.__getitem__(index)[0].size())
    print(train_data.__getitem__(index)[1])
    
    
def test3():
    train_dataloader, test_dataloader = createDataLoader()
    iterator = iter(train_dataloader ) #dataloaders_dict["train"])  # イテレータに変換
    inputs, labels = next(iterator)  # 1番目の要素を取り出す
    print(inputs.size())
    print(labels)
    
if __name__ == "__main__":
    main()