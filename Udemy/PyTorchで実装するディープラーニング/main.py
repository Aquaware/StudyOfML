import torch
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn

def learnTensor1():
    a = torch.tensor([1, 2, 3])
    print(a, type(a))
    b = torch.tensor([1, 2, 3],dtype=torch.float64)
    print(b)
    c = torch.arange(1, 5)
    print(c)
    d = torch.rand(10)
    print(d)
    e = torch.zeros(5)
    print(e)
    f = torch.tensor([[-1, 2, 3, 4], [6, 8, -9.0, 10.0]])
    print(f, f.size())
    g = f.numpy()
    print(g)

    h = f[0, 0:3]
    print(h)
    j = f[:, 0:2]
    print(j)

    f[f < 0] = 0
    print(f)

def learnTensor2():
    a = torch.tensor([1.2, np.nan, 3.4])
    b = torch.tensor([4, 5, 6])

    c = a + b
    print(c)

    d = a * b
    print(d)

    print(a.mean())

def recognizeHandeWrittenFigures():
    data_set = datasets.load_digits()
    data = data_set.data
    target = data_set.target
    rows = 2
    cols = 5
    fig = plt.figure()
    for row in range(rows):
        for col in range(cols):
            i = row * cols + col
            image = data[i]
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(image.reshape(8,8), cmap='Greys_r')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(target[i])

    train_x, test_x, train_y, test_y = train_test_split(data, target)


    tensor_train_x = torch.tensor(train_x, dtype = torch.float32)
    tensor_test_x = torch.tensor(test_x, dtype = torch.float32)
    tensor_train_y = torch.tensor(train_y, dtype=torch.int64)
    tensor_test_y = torch.tensor(test_y, dtyp=torch.int64)
    
    net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, 16), nn.ReLU(),
                        nn.Linear(16, 10))
    
    loss_function = nn.CrossEntropyLoss()
    solver = torch.optim.SGD(net.parameters(), lr=0.01)
    
    
    


if __name__ == '__main__':
    recognizeHandeWrittenFigures()


