import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms
from ImageTransform import ImageTransform
from ImageClassPredictor import ImageClassPredictor

print('Pytorch Version', torch.__version__)
print('Torchvision version', torchvision.__version__)

def main():

    file_path = './data/goldenretriever-3724972_640.jpg'
    image = Image.open(file_path)
    #plt.imshow(image)
    #plt.show()
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(size, mean, std)
    transformed_image = transform(image)
    #plt.imshow(torchImage2PIL(transformed_image))
    #plt.show()

    class_index = json.load(open('./data/imagenet_class_index.json'))
    predictor = ImageClassPredictor(class_index)

    net_inputs = transformed_image.unsqueeze_(0)
    net = loadModel()
    net_out = net(net_inputs)
    result = predictor.predict(net_out)
    print(result)

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



