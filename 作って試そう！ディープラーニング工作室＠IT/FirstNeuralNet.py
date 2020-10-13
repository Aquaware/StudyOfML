
import torch
from torch import nn

layers_num = [4, 5, 1]

class FirstNeuralNet(nn.Module):

    def __init__(self):
        super().__init__(self)
        self.layer1 = nn.Linear(input_layer_num, hidden_layer_num)
        self.layer2 = nn.Linear(hidden_layer_num, output_layer_num)

    def forward(self, x):
        x = self.affine1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    net = FirstNeuralNet()
