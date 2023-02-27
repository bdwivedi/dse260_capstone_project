from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
from models.convlstm_wind import EncoderRNN, ConvLSTM
from torch.autograd import Variable
# from penalty import DivergenceLoss
from utils_convlstm_wind import train_epoch, eval_epoch, test_epoch
from data.dataset_attension import IdealizedGrasslands
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

train_direc="/root/autodl-tmp/data/"
test_direc="/root/autodl-tmp/data/"

min_mse=10
output_length=100
input_length=10
learning_rate=0.001
dropout_rate=0
kernel_size=3
batch_size=1

train_indices=list(range(0,500))
valid_indices = list(range(550, 600))
test_indices = list(range(600, 650))
convcell =  ConvLSTM(input_shape=(75,75), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)
model= EncoderRNN(convcell,device=device).to(device)
train_set = IdealizedGrasslands(train_indices, input_length , 15, output_length, train_direc)
valid_set = IdealizedGrasslands(valid_indices, input_length , 15, output_length, test_direc)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)
loss_fun = torch.nn.L1Loss()
#regularizer = DivergenceLoss(torch.nn.MSELoss())

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 5e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.9)


train_mse = []
valid_mse = []
test_mse = []
for i in range(50):
    start = time.time()
    torch.cuda.empty_cache()
    scheduler.step()
    model.train()
    teacher_force_ratio=np.maximum(0, 1 - i * 0.03)
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun,teacher_force_ratio))
    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1]
        best_model = model
        torch.save(best_model, "convlstm_wind_model.pth")
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5))