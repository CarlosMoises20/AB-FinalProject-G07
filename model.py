
from dataLoad import DataLoad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# https://www.almabetter.com/bytes/articles/image-classification-using-pytorch

    # Starting from 3. Define the neural network



# develop, train, test and validate the model


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

def train_network(num_epochs=100, learning_rate=0.70, momentum=0.9):
    # Load and preprocess the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainloader,testloader,_  = DataLoad.load_data()

    # Initialize the network, criterion, and optimizer
    net = Net()
    criterion, optimizer = define_loss_momentum(learning_rate, momentum)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = net(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize the weights

            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model

    return net, trainloader , testloader




# testing


net, trainloader, testloader = train_network(num_epochs=100, learning_rate=0.700, momentum=0.9, batch_size=4)

print(trainloader)


# validate




# testing




# validate

