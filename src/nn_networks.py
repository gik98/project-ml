import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional
import torch.utils.data as data
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class FullyConnectedNN2(nn.Module):
    def __init__(self, channels, num_classes, activation_type=nn.ReLU):
        super(FullyConnectedNN2, self).__init__()
        self.num_classes = num_classes
        
        activation = activation_type()
        self.drop = nn.Dropout(p = 0.2)

        # We create the list of layers in the order we want them to be evaluated
        layers = []
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i-1], channels[i]))
            layers.append(activation)
            if i%2:
                layers.append(self.drop)

        layers.append(nn.Linear(channels[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def save(self, path):
        with open(path, "wb") as file:
            torch.save(self, file)


def NN_load(path):
    ret = torch.load(path)
    return ret


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.num_classes = 4
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(27 * 27 * 16, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(10, 4)


    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 1*27*27*16).to(device)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AugmentedConvNN(nn.Module):
    def __init__(self):
        super(AugmentedConvNN, self).__init__()
        self.num_classes = 4
        
        self.pool = nn.MaxPool2d(3, 3)
        self.drop = nn.Dropout(p = 0.2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, 3)
        #self.fc1 = nn.Linear(27 * 27 * 16, 256)
        #self.fc2 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(27*27*16, 64)

        self.fc3a = nn.Linear(64, 42)
        self.fc3b = nn.Linear(84, 42)
        
        self.fc4 = nn.Linear(84, 42)
        self.fc5 = nn.Linear(42, 10)
        self.fc6 = nn.Linear(10, 4)

        self.norm = transforms.Normalize(0.5, 0.5)

    def forward(self, X):
        x = X[:,:,:,:256]
        y = X[:,:,:,256]

        y = torch.squeeze(y)
        y = y[:,:84]

        x = self.pool(self.drop(functional.relu(self.conv1(x))))
        x = self.pool(self.drop(functional.relu(self.conv2(x))))
        x = x.view(-1, 1*27*27*16).to(device)
        x = functional.relu(self.fc1(x))
        #x = functional.relu(self.fc2(x))
        
        x = functional.relu(self.fc3a(x))
        y = functional.relu(self.fc3b(y))

        x = self.norm(x.unsqueeze(0)).squeeze()
        y = self.norm(y.unsqueeze(0)).squeeze()

        x = torch.cat((x, y), dim=1)

        x = self.fc4(self.drop(x))
        x = self.fc5(x)
        x = self.fc6(x)
        return x
    
    def save(self, path):
        with open(path, "wb") as file:
            torch.save(self, file)
        