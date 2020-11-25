import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, transforms
from ImageTransform import ImageTransform
from ImageClassPredictor import ImageClassPredictor
from DataSet import DataSet
import tqdm

print('Pytorch Version', torch.__version__)
print('Torchvision version', torchvision.__version__)

def main():
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    batch_size = 32

    dir_path = './data/train/'
    train_data = DataSet('train', size, mean, std, dir_path)
    train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    dir_path = './data/test/'
    test_data = DataSet('test', size, mean, std, dir_path)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    data_loader = {'train': train, 'test': test}
    iterator = iter(data_loader['train'])
    inputs, labels = next(iterator)
    print(inputs.size())
    print(labels)

    net = loadModel()
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
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

def torchImage2PIL(image_data):
    image = image_data.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    return image



def loadModel():
    net = models.vgg16(pretrained=True)
    net.eval()
    #print(net)
    print('VGG16 model loaded')
    return net

if __name__ == "__main__":
    main()



