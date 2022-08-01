import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# Model definition
class WideNet(nn.Module):
  def __init__(self, alpha=50): # alpha eq neuron number of the hiden layer
    super().__init__()

#    self.fc1 = nn.Linear(1034,50)
    self.fc1 = nn.Linear(1033, alpha)
    self.fc2 = nn.Linear(alpha, 1)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# Model CNN definition
class CNNNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(
        in_channels = 1, out_channels = 17,
        kernel_size = 3, padding = 1 )
    self.conv2 = nn.Conv2d(
        in_channels = 17, out_channels = 17,
        kernel_size = 3, padding = 1 )
    self.conv3 = nn.Conv2d(
        in_channels = 17, out_channels = 17,
        kernel_size = 3, padding = 1 )
    # self.conv4 = nn.Conv2d(
    #     in_channels = 17, out_channels = 17,
    #     kernel_size = 3, padding = 1 )
    # self.conv5 = nn.Conv2d(
    #     in_channels = 17, out_channels = 17,
    #     kernel_size = 3, padding = 1 )
    #self.pool1 = nn.MaxPool2d(kernel_size = 3)
    self.pool1 = nn.AvgPool2d(kernel_size = 2)
    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(p = 0.5)
    self.fc1 = nn.Linear(3774, 60)
    self.fc2 = nn.Linear(60, 1)
    # self.fc1 = nn.Linear(1470, 1)
    #self.fc1 = nn.Linear(3330, 1)
    #self.fc1 = nn.Linear(1666, 1)
    

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    # x = F.relu(self.conv4(x))
    # x = F.relu(self.conv5(x))
    x = self.pool1(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# model_CNN = CNNNet().to(device)

# model_CNN(torch.rand(64,1,148,7).to(device)).shape

# Model WIDE&CNN definition
class WIDECNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(2, 1)

  def forward(self, x):
    x = self.fc1(x)
    return x

# model_WIDECNN = WIDECNN().to(device)

