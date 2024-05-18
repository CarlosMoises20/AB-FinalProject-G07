import torch

from pneumoniaDetector import *


data = [[1, 2, 3, 4], [3, 4, 5, 6]]

x_data = torch.tensor(data)
print(x_data)