from dataLoad import DataLoad
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

# Develop, train, test and validate the model


class Model:
    def __init__(self):
        self.__model = models.resnet18(pretrained=True)
        self.__criterion = nn.CrossEntropyLoss()
        self.__data = DataLoad()
        self.__trainloader, self.__testloader, self.__valloader = self.__data.load_data()

    # Training
    def train_model(self, num_epochs, learning_rate, momentum, weight_decay=0.0005):
        optimizer = optim.SGD(self.__model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        total_loss = []
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            running_loss = 0.0
            self.__model.train()
            for i, data in enumerate(self.__trainloader, 0):
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
            val_acc, _, _ = self.evaluate_model(self.__valloader)
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
            for data in dataloader:
                images, labels = data
                outputs = self.__model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_pred_list.extend(predicted.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())            

        accuracy = 100 * correct / total
        return accuracy, classification_report(y_true_list, y_pred_list), confusion_matrix(y_true_list, y_pred_list)


    def test_model(self):
        test_accuracy, classification_report, confusion_matrix = self.evaluate_model(self.__testloader)
        return test_accuracy, classification_report, confusion_matrix