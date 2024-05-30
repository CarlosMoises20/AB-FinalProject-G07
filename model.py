
from dataLoad import DataLoad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

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
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x







class Model:
    def __init__(self):
        self.__model = Net()
        #self.__criterion = nn.CrossEntropyLoss()
        #self.__optimizer = optim.SGD(self.__model.parameters(), lr=0.001, momentum=0.9)
        self.__data = DataLoad()
        self.__trainloader, self.__testloader, self.__valloader = self.__data.load_data()
        #self.__transform = transforms.Compose(
        #    [transforms.ToTensor(),
        #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        

    def __define_loss_momentum(self, loss=0.001, momentum=0.9):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.__model.parameters(), lr=loss, momentum=momentum)
        return criterion, optimizer

    # training

    def train_network(self, num_epochs=100, learning_rate=0.7, momentum=0.9):
        
        criterion, optimizer = self.__define_loss_momentum(learning_rate, momentum)

        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.__trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()  # Zero the parameter gradients

                outputs = self.__model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize the weights

                running_loss += loss.item()
                if i % 2000 == 1999:  # Print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

        # Save the trained model




model = Model()
model.train_network(num_epochs=100, learning_rate=0.700, momentum=0.9)

#print(trainloader)

