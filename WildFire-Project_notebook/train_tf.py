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
from models.tfnet import LES
from torch.autograd import Variable
# from penalty import DivergenceLoss
from utils_tfnet import train_epoch, eval_epoch, test_epoch
from data.dataset import IdealizedGrasslands
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

train_direc="/root/autodl-tmp/data/"
test_direc="/root/autodl-tmp/data/"
min_mse=10
time_range=5
output_length=100
input_length=14
learning_rate=0.001
dropout_rate=0
kernel_size=3
batch_size=1

train_indices=list(range(0,850))
valid_indices = list(range(850, 950))
test_indices = list(range(950, 1150))
model=LES(input_channels = input_length, output_channels = 1, kernel_size = kernel_size,
            dropout_rate = dropout_rate, time_range = time_range).to(device)
train_set = IdealizedGrasslands(train_indices, input_length+time_range-1 , 21, output_length, train_direc)
valid_set =IdealizedGrasslands(valid_indices, input_length+time_range-1 , 21, output_length, test_direc)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)
loss_fun = torch.nn.L1Loss()
#regularizer = DivergenceLoss(torch.nn.MSELoss())

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas = (0.9, 0.999), weight_decay = 4e-4)
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
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun, teacher_force_ratio))#
    model.eval()
    mse, preds, trues = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)
    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1]
        best_model = model
        torch.save(best_model, "tfnet_model.pth")
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(train_mse[-1], valid_mse[-1], round((end-start)/60,5))



