
import os, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim


class PneumoniaDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )




data_dir = '../Chest_X-Ray_Dataset'
train_dir = 'train'
test_dir = 'test'
val_dir = 'val'

classes = ('NORMAL', 'PNEUMONIA')