import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms

class ImageTransform():

    def __init__(self, size, mean, std):
        self.transform = transforms.Compose([transforms.Resize(size),
                                             transforms.CenterCrop(size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])

    def __call__(self, image):
        return self.transform(image)

