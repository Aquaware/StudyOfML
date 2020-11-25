import glob
import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.utils.data as data
import torch.optim as optim
from torchvision import models, transforms
from ImageTransform import ImageTransform
from ImageClassPredictor import ImageClassPredictor

BEE = 'bee'
ANT = 'ant'
LABEL = {ANT: 0, BEE: 1}

class DataSet(data.Dataset):
    def __init__(self, phase, size, mean, std, dir_path):
        self.phase = phase
        self.size = size
        self.mean = mean
        self.std = std
        self.dir_path = dir_path
        self.file_list = self.fileList(dir_path)
        print(self.file_list)

    def fileList(self, dir_path):
        out = []
        for holder, subholders, files in os.walk(dir_path):
            for file in files:
                out.append(holder + '/' + file)
        return out

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        transform = ImageTransform(self.size, self.mean, self.std)
        path = self.file_list[index]
        image = Image.open(path)
        data = transform(image)
        if path.find('ants') >= 0:
            target = LABEL[ANT]
        elif path.find('bees') >= 0:
            target = LABEL[BEE]
        return (data, target)
