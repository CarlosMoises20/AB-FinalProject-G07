
import os, torch, torchvision
from torchvision import datasets, transforms
import torch.nn as nn
from pathlib import Path        # To get the name of a file
from torch.utils.data import DataLoader


# https://www.almabetter.com/bytes/articles/image-classification-using-pytorch

    # 1 e 2


# Algoritmo de classificação de imagens utilizado para desenvolver o modelo
class ImageClassifier(nn.Module):

    """
    data_dir -> the data root path
    train_dir -> the directory for the images used to train the model
    test_dir -> the directory for the images used to test the model
    val_dir -> the directory for the images used to validate the model
    classes -> labels to classify images
                - all images on NORMAL folders are classified as NORMAL
                - the other images are classified either as VIRUS or as BACTERIA, depending of the image's name

    """
    def __init__(self, train_dir = './chest_xray/train', 
                 test_dir = './chest_xray/test', 
                 val_dir = './chest_xray/val'):
        
        self.__test_dir = test_dir
        self.__train_dir = train_dir
        self.__val_dir = val_dir
        self.__classes = ['NORMAL', 'VIRUS', 'BACTERIA']


    def load_data(self):

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        # Load the datasets with ImageFolder
        trainset = datasets.ImageFolder(root=self.__train_dir, transform=transform)
        valset = datasets.ImageFolder(root=self.__val_dir, transform=transform)
        testset = datasets.ImageFolder(root=self.__test_dir, transform=transform)

        print(trainset)

        #trainset.samples = [(path, self.__get_label(os.path.basename(path))) for path, _ in trainset.samples]
        #valset.samples = [(path, self.__get_label(os.path.basename(path))) for path, _ in valset.samples]
        #testset.samples = [(path, self.__get_label(os.path.basename(path))) for path, _ in testset.samples]

        # Create DataLoaders for each dataset
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
        valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=2)
        testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

        """
        trainset = torchvision.datasets.CIFAR10(root = self.__train_dir, train=True,
                                                download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root = self.__test_dir, train=False,
                                            download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=2)
        
        valset =  torchvision.datasets.CIFAR10(root = self.__val_dir, train=False,
                                                download=True, transform=transform)
        
        valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                                shuffle=False, num_workers=2)
        """
        

        return trainloader, testloader, valloader
    

    def get_class_names(self):
        return self.__classes
    

    def __get_label(file_name):
        if 'virus' in file_name.lower():
            return 'VIRUS'
        elif 'bacteria' in file_name.lower():
            return 'BACTERIA'
        else:
            return 'NORMAL'
        

load_data = ImageClassifier()

print(load_data.load_data())



