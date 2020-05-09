import torch
import torch.nn as nn
import torch.optim as optim


class FullyConnectedNN(nn.Module):
    def __init__(self, channels, num_classes, activation_type=nn.ReLU):
        super(FullyConnectedNN, self).__init__()
        self.num_classes = num_classes

        activation = activation_type()

        # We create the list of layers in the order we want them to be evaluated
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i-1], channels[i]))
            layers.append(activation)

        layers.append(nn.Linear(channels[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def save(self, path):
        with open(path, "wb") as file:
            torch.save(self, file)

def NN_load(path):
    ret = None
    with open(path, "rb") as file:
        ret = torch.load(path)
    
    return ret