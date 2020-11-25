import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms

class ImageTransform():

    def __init__(self, size, mean, std):
        at_train = transforms.Compose(      [transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

        at_evaluate = transforms.Compose(      [transforms.Resize(size),
                                             transforms.CenterCrop(size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
        self.transform = {'train': at_train, 'test': at_evaluate}

    def __call__(self, image, phase='train'):
        try:
            transform = self.transform[phase.lower()]
            return transform(image)
        except:
            return None
