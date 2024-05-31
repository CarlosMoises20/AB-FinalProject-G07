from dataLoad import DataLoad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm.notebook import tqdm
import seaborn as sns
import numpy as np

# Develop, train, test and validate the model

class Net(nn.Module):
    def __init__(self, num_classes=2):
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
        self.__model = models.resnet18(pretrained=True)
        self.__criterion = nn.CrossEntropyLoss()
        self.__data = DataLoad()
        self.__trainloader, self.__testloader, self.__valloader = self.__data.load_data()
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training
    def train_model(self, num_epochs, learning_rate, momentum, weight_decay=0.0005):
        optimizer = optim.SGD(self.__model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        total_loss = []
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.__model.train()
            for _, data in enumerate(self.__trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.__model(inputs)
                loss = self.__criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(self.__trainloader)
            total_loss.append(epoch_loss)
            
            # Avaliação no conjunto de validação
            val_acc = self.evaluate_model(self.__valloader)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        print('Finished Training')
        return total_loss, best_val_acc

    def evaluate_model(self, dataloader):
        self.__model.eval()
        correct = 0
        total = 0
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            """
            for x_batch, y_batch in tqdm(dataloader):
                x_batch, y_batch = x_batch.to(self.__device), y_batch.to(self.__device)
                y_test_pred = self.__model(x_batch)
                _, y_pred_tag = torch.max(y_test_pred, dim = 1)
                y_pred_list.append(y_pred_tag.cpu().numpy())
                y_true_list.append(y_batch.cpu().numpy())
            """
            
            for data in dataloader:
                images, labels = data
                outputs = self.__model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_pred_list.extend(predicted.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())
            

        accuracy = 100 * correct / total
        return accuracy


    def test_model(self):
        test_accuracy = self.evaluate_model(self.__testloader)
        return test_accuracy