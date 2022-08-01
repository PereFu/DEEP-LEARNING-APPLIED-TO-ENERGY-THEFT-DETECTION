import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# CNN

# De numpy a torch tensor y creación del dataset
def CNNDataset_func (df_train_norm_CNN, df_test_norm_CNN, df_flag_train_CNN, df_flag_test_CNN):

  df_train_CNN_list = []
  df_test_CNN_list = []

  m = torch.zeros((148, 7))
  tril_indices = torch.tril_indices(row=148, col=7, offset=6)
  zeros = torch.zeros(3)

  for i in range(df_train_norm_CNN.shape[0]):
    aux = torch.cat((df_train_norm_CNN[i], zeros), 0)
    m[tril_indices[0], tril_indices[1]] = aux
    df_train_CNN_list.append(m)
    m = torch.zeros((148, 7))

  for i in range(df_test_norm_CNN.shape[0]):
    aux = torch.cat((df_test_norm_CNN[i], zeros), 0)
    m[tril_indices[0], tril_indices[1]] = aux
    df_test_CNN_list.append(m)
    m = torch.zeros((148, 7))

  df_train_CNN_stack = torch.stack(df_train_CNN_list)
  df_train_CNN_stack = df_train_CNN_stack[:,None, :] # 1 channel
  df_test_CNN_stack = torch.stack(df_test_CNN_list)
  df_test_CNN_stack = df_test_CNN_stack[:,None, :]# 1 channel

  return df_train_CNN_stack, df_test_CNN_stack, df_flag_train_CNN, df_flag_test_CNN


def g1(x):
  w = torch.tensor([[ # shape: (Cin, Cout, kH, hW) = (1, 1, 3, 3)
      [[2, -1, -1],
       [2, -1, -1],
       [2, -1, -1]]
  ]], dtype=torch.float32)
  return F.conv2d(x, w, stride=1, padding=1)

def g2(x):
  w = torch.tensor([[
      [[ 2,  2,  2],
       [-1, -1, -1],
       [-1, -1, -1]]
  ]], dtype=torch.float32)
  return F.conv2d(x, w, stride=1, padding=1)


# g1+g2 calculation in getitem
class CNNDataset(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return self.y.shape[0]

  def __getitem__(self, idx):
    return g1(self.x[idx]) + g2(self.x[idx]), self.y[idx]

# WIDE&CNN

# Dataset class
class JoinDataset(Dataset):
  def __init__(self, ds1, ds2):
    self.ds1 = ds1
    self.ds2 = ds2

  def __len__(self):
    return self.ds1.__len__()
  def __getitem__(self, idx):
    return self.ds1[idx][1], self.ds1[idx][0], self.ds2[idx][0]
