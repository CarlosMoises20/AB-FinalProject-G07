from dataLoad import DataLoad
import torch
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
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(59536, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x




class Model:
    def __init__(self):
        self.__model = Net()
        self.__criterion = nn.CrossEntropyLoss()
        self.__data = DataLoad()
        self.__trainloader, self.__testloader, _ = self.__data.load_data()


    # training

    def train_network(self, num_epochs, learning_rate, momentum):
        
        """
        params = iterable of parameters to optimize or dicts defining parameter groups
        lr = learning rate
        momentum = momentum of the gradient update
        
        """
        optimizer = optim.SGD(params=self.__model.parameters(), lr=learning_rate, momentum=momentum)
        total_loss = []

        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for _, data in enumerate(self.__trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()  # Zero the parameter gradients

                outputs = self.__model(inputs)  # Forward pass
                loss = self.__criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                optimizer.step()  # Optimize the weights

                running_loss += loss.item()

            epoch_loss = running_loss / len(self.__trainloader)
            total_loss.append(epoch_loss)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

        print('Finished Training')

        # Save the trained model
        return total_loss
    
    def test_network(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.__testloader:
                images, labels = data
                outputs = self.__model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        return test_accuracy


"""

Auxiliar code to test this module
if __name__ == '__main__':
    # run the model to see the results
    multiprocessing.freeze_support()

    model = Model()
    
    n_epochs = 15
    
    total_loss = model.train_network(num_epochs=n_epochs, learning_rate=0.85, momentum=0.9)
    
    # Plot the loss evolution, where the x-axis is the epoch and the y-axis is the loss
    plt.plot(range(1, n_epochs + 1), total_loss)
    plt.xticks(range(1, n_epochs + 1))  # Set the x-axis ticks to 1, 2, 3
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.show()

    test_accuracy = model.test_network()
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    model.save_model("./models")
    
"""
