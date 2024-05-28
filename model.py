
from dataLoader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://www.almabetter.com/bytes/articles/image-classification-using-pytorch

    # A partir de 3. Define the neural network



# desenvolver, treinar, testar e validar o modelo neste ficheiro


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def define_loss_momentum(loss=0.001, momentum=0.9):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=loss, momentum=momentum)
    return criterion, optimizer



# training





# testing




# validate

