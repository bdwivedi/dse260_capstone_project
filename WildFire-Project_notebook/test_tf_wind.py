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
from models.tfnet_wind import LES
from torch.autograd import Variable
from utils_tf_wind import train_epoch, eval_epoch, test_epoch
from data.dataset_attension import IdealizedGrasslands
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

train_direc="/root/autodl-tmp/data/"
test_direc="/root/autodl-tmp/data/"

min_mse=10
output_length=200
input_length=7
time_range=3
learning_rate=0.001
dropout_rate=0
kernel_size=3
batch_size=1

train_indices=list(range(0,500))
valid_indices = list(range(550, 600))
test_indices = list(range(600, 650))
loss_fun = torch.nn.L1Loss()
#loss_local=localLoss()
#DL=decreaseLoss()
best_model = torch.load("tfnet_wind_model.pth")
test_set = IdealizedGrasslands(test_indices, input_length+time_range-1 , 15, output_length, test_direc)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
loss_curve,preds, trues  = test_epoch(test_loader, best_model, loss_fun)
print(preds.shape)
print(trues.shape)


torch.save({"preds": preds[:10],
            "trues": trues[:10],
            "loss_curve": loss_curve},
            "/root/autodl-tmp/tfnet_wind_results.pt")
