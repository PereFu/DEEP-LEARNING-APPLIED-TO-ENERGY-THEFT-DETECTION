import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import sklearn

import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib as mpl 
import matplotlib.pyplot as plt
from tqdm import tqdm # Execution progress
import numpy as np

# Torch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset

# sci-kit learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

# Others
from mlxtend.plotting import plot_confusion_matrix
from IPython.core.debugger import set_trace


loss_fn = nn.BCEWithLogitsLoss()
# GPU as device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

# Data normalitzation (desv estandard and consumer mean)
def normalizacion (x_train_fold, x_test_fold, y_train_fold, y_test_fold):
  df_train_norm = (x_train_fold - np.mean(x_train_fold)) / np.std(x_train_fold)
  df_test_norm = (x_test_fold - np.mean(x_test_fold)) / np.std(x_test_fold)
  df_flag_train = y_train_fold
  df_flag_test = y_test_fold

  # Creación de dataset y conversión de numpy array a torch tensor
  df_train_norm = torch.from_numpy(df_train_norm.to_numpy()).float()
  df_test_norm = torch.from_numpy(df_test_norm.to_numpy()).float()
  df_flag_train = torch.from_numpy(df_flag_train.to_numpy()).float()
  df_flag_test = torch.from_numpy(df_flag_test.to_numpy()).float()
  return df_train_norm, df_test_norm, df_flag_train, df_flag_test

# Train
def train (dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  train_loss = 0
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device)

    # Prediction error calcul
    pred = model(X)
    loss = loss_fn(pred, y)
    train_loss += loss.item()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # loss, current = loss.item(), batch * len(X)
  # print(f"TRAIN LOSS: {loss:>7f} [{current:>5d}/{size:>5d}]")
  print("TRAIN LOSS:", 1000*train_loss/size)

# Test
def test_inference(dataloader, model):
  size = len(dataloader.dataset)
  model.eval()
  preds, ys = [], []
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      preds.append(torch.sigmoid(pred).cpu())
      ys.append(y.cpu())
      
  ps = torch.concat(preds)
  ys = torch.concat(ys)
  return ps, ys

def test_validate(dataloader, model):
  size = len(dataloader.dataset)
  model.eval()
  test_loss, rocauc = 0, 0
  preds, ys = [], []
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      preds.append(torch.sigmoid(pred).cpu())
      ys.append(y.cpu())
      # rocauc += sklearn.metrics.roc_auc_score(y.cpu(),torch.sigmoid(pred).cpu())
      # correct += (pred.argmax(1) == y).type(torch.float).sum().item() # Arreglar, esto es para multiclase

  test_loss /= size
  ps = torch.concat(preds)
  ys = torch.concat(ys)
  print("TEST METRICS:", test_loss*1000, sklearn.metrics.roc_auc_score(ys,ps))

# Train
def train_WIDECNN (dataloader, model_wide, model_cnn, model_widecnn, loss_fn, optimizer):
  size = len(dataloader.dataset)
  train_loss_w = 0
  train_loss_c = 0
  train_loss_wc = 0
  # pred_wcs = []

  for batch, (y_w, X_w, X_c) in enumerate(dataloader):
    X_w, X_c, y_w = X_w.to(device), X_c.to(device), y_w.to(device)
    # Prediction calcul error
    pred_w = model_wide(X_w)
    pred_c = model_cnn(X_c)
    # pred_wc = pred_w + pred_c # Method 1
    pred_w_c = torch.stack((pred_w, pred_c)).swapaxes(0,1).squeeze(2) # Method 2
    pred_wc = model_widecnn(pred_w_c) # Method 2

    # WIDECNN
    loss_wc = loss_fn(pred_wc, y_w)
    train_loss_wc += loss_wc.item()

    # Backpropagation
    optimizer.zero_grad()
    loss_wc.backward()
    optimizer.step()

  print("TRAIN LOSS:", 1000*train_loss_wc/size)

  # Test
def test_inference_WIDECNN(dataloader, model_wide, model_cnn, model_widecnn):
  size = len(dataloader.dataset)
  model_wide.eval()
  model_cnn.eval()
  model_widecnn.eval()

  preds, ys = [], []
  with torch.no_grad():
    for y_w, X_w, X_c in dataloader:
      X_w, X_c, y_w = X_w.to(device), X_c.to(device), y_w.to(device)
      pred_w = model_wide(X_w)
      pred_c = model_cnn(X_c)
      # pred_wc = pred_w + pred_c # Method 1
      pred_w_c = torch.stack((pred_w, pred_c)).swapaxes(0,1).squeeze(2) # Method 2
      pred_wc = model_widecnn(pred_w_c) # Method 2
      preds.append(torch.sigmoid(pred_wc).cpu())
      ys.append(y_w.cpu())
      
      
  ps = torch.concat(preds)
  ys = torch.concat(ys)
  return ps, ys

def test_validate_WIDECNN(dataloader, model_wide, model_cnn, model_widecnn):
  size = len(dataloader.dataset)
  model_wide.eval()
  model_cnn.eval()
  model_widecnn.eval()
  test_loss, rocauc = 0, 0
  preds, ys = [], []

  with torch.no_grad():
    for y_w, X_w, X_c in dataloader:
      X_w, X_c, y_w = X_w.to(device), X_c.to(device), y_w.to(device)
      pred_w = model_wide(X_w)
      pred_c = model_cnn(X_c)
      # pred_wc = pred_w + pred_c # Method 1
      pred_w_c = torch.stack((pred_w, pred_c)).swapaxes(0,1).squeeze(2) # Method 2
      pred_wc = model_widecnn(pred_w_c) # Method 2
      preds.append(torch.sigmoid(pred_wc).cpu())
      ys.append(y_w.cpu())
      test_loss += loss_fn(pred_wc, y_w).item()

  test_loss /= size
  ps = torch.concat(preds)
  ys = torch.concat(ys)
  print("TEST METRICS:", test_loss*1000, sklearn.metrics.roc_auc_score(ys,ps))


