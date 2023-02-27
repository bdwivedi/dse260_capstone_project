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
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
#import kornia
warnings.filterwarnings("ignore")

def merge(x_fuel,x_windu,x_windv):
    l=[]
    for j in range(x_fuel.shape[1]):
        x=torch.cat((x_fuel[:,j].unsqueeze(1),x_windu[:,j].unsqueeze(1),
                     x_windv[:,j].unsqueeze(1)),dim=1)
        l.append(x)
    return torch.cat(l,dim=1)


def train_epoch(train_loader, model, optimizer, loss_function, teacher_force_ratio):
    train_mse = []
    for x_fuel,x_windu,x_windv, y_fuel,y_windu,y_windv in train_loader:

        loss = 0
        ims = []
        x_fuel = x_fuel.to(device)
        y_fuel = y_fuel.to(device)
        x_windu=x_windu.to(device)
        x_windv=x_windv.to(device)
        y_windu=y_windu.to(device)
        y_windv=y_windv.to(device)
        
        x_fuel=torch.squeeze(x_fuel,dim=2)
        y_fuel=torch.squeeze(y_fuel,dim=2)
        x_windu=torch.squeeze(x_windu,dim=2)
        x_windv=torch.squeeze(x_windv,dim=2)
        y_windu=torch.squeeze(y_windu,dim=2)
        y_windv=torch.squeeze(y_windv,dim=2)
        
        use_teacher_force=True if np.random.random()<teacher_force_ratio else False
        for i in range(y_fuel.shape[1]):
            y=y_fuel[:,i]
            windu=y_windu[:,i]
            windv=y_windv[:,i]
            
                
            xx=merge(x_fuel,x_windu,x_windv) #b*c*h*w
            im = model(xx)
            if use_teacher_force:
                x_fuel=torch.cat([x_fuel[:,1:],y.unsqueeze(1)],1)
            
            else:
                x_fuel=torch.cat([x_fuel[:,1:],im],1)
             
            x_windu=torch.cat([x_windu[:,1:],windu.unsqueeze(1)],dim=1)
            x_windv=torch.cat([x_windv[:,1:],windv.unsqueeze(1)],dim=1)

            xx=merge(x_fuel,x_windu,x_windv)
            loss += loss_function(im, y)
           
            #ims.append(im.cpu().data.numpy())

        #ims = np.concatenate(ims, axis=1)
        train_mse.append(loss.item() / y_fuel.shape[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mse = round((np.mean(train_mse)), 5)
    return train_mse


def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for x_fuel,x_windu,x_windv, y_fuel,y_windu,y_windv in valid_loader:
        # print(xx.size())
        # print(yy.size())
            loss = 0
            ims = []
            x_fuel = x_fuel.to(device)
            y_fuel = y_fuel.to(device)
            x_windu=x_windu.to(device)
            x_windv=x_windv.to(device)
            y_windu=y_windu.to(device)
            y_windv=y_windv.to(device)
            x_fuel=torch.squeeze(x_fuel,dim=2)
            y_fuel=torch.squeeze(y_fuel,dim=2)
            x_windu=torch.squeeze(x_windu,dim=2)
            x_windv=torch.squeeze(x_windv,dim=2)
            y_windu=torch.squeeze(y_windu,dim=2)
            y_windv=torch.squeeze(y_windv,dim=2)
       # print(yy.shape)
           
            for i in range(y_fuel.shape[1]):
                y=y_fuel[:,i]
                windu=y_windu[:,i]
                windv=y_windv[:,i]
                xx=merge(x_fuel,x_windu,x_windv) #b*c*h*w
                im = model(xx)
    
                x_fuel=torch.cat([x_fuel[:,1:],im],1)         
                x_windu=torch.cat([x_windu[:,1:],windu.unsqueeze(1)],dim=1)
                x_windv=torch.cat([x_windv[:,1:],windv.unsqueeze(1)],dim=1)
                xx=merge(x_fuel,x_windu,x_windv)
                loss += loss_function(im, y)
            valid_mse.append(loss.item() / y_fuel.shape[1])

        valid_mse=round((np.mean(valid_mse)), 8)
    return valid_mse, preds, trues


def test_epoch(test_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for x_fuel,x_windu,x_windv, y_fuel,y_windu,y_windv in test_loader:

            loss = 0
            ims = []
            x_fuel = x_fuel.to(device)
            y_fuel = y_fuel.to(device)
            x_windu=x_windu.to(device)
            x_windv=x_windv.to(device)
            y_windu=y_windu.to(device)
            y_windv=y_windv.to(device)
            x_fuel=torch.squeeze(x_fuel,dim=2)
            y_fuel=torch.squeeze(y_fuel,dim=2)
            x_windu=torch.squeeze(x_windu,dim=2)
            x_windv=torch.squeeze(x_windv,dim=2)
            y_windu=torch.squeeze(y_windu,dim=2)
            y_windv=torch.squeeze(y_windv,dim=2)
            loss = 0
            ims = []

            for i in range(y_fuel.shape[1]):
                y=y_fuel[:,i]
                windu=y_windu[:,i]
                windv=y_windv[:,i]
                xx=merge(x_fuel,x_windu,x_windv) #b*c*h*w
                im = model(xx)
                x_fuel=torch.cat([x_fuel[:,1:],im],1)
                x_windu=torch.cat([x_windu[:,1:],windu.unsqueeze(1)],dim=1)
                x_windv=torch.cat([x_windv[:,1:],windv.unsqueeze(1)],dim=1)
                xx=merge(x_fuel,x_windu,x_windv) 
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())

                ims.append(im.cpu().data.numpy())

            ims = np.array(ims).transpose(1, 0, 2, 3, 4)
            preds.append(ims)
            trues.append(y_fuel.cpu().data.numpy())
            valid_mse.append(loss.item() / y_fuel.shape[1])

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        prediction_length=preds.shape[1]
        loss_curve=np.array(loss_curve).reshape(-1, prediction_length)
        loss_curve=(np.mean(loss_curve,axis=0))
    return loss_curve,preds, trues


