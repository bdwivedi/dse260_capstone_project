import torch.nn as nn
import os
import numpy as np
import argparse
import torch
import errno
from torch.optim.lr_scheduler import ReduceLROnPlateau
from numpy import *
from numpy.linalg import *
from scipy.special import factorial
from functools import reduce
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from functools import reduce


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_epoch(train_loader, model, optimizer, loss_function, teacher_force_ratio):
    train_mse = []
    for xx, yy in train_loader: #batch_size*time_step*channel*height*width
        input_length=xx.size(1)
        output_length=yy.size(1)
        loss = 0
        ims = []
        xx = xx.to(device)
        yy = yy.to(device)
        hidden=None
        for i in range(input_length-1):
            hidden,output_image=model(xx[:,i],i==0,hidden=hidden)

        use_teacher_force=True if np.random.random()<teacher_force_ratio else False
        if(use_teacher_force):
            decoder_input=xx[:,-1,:,:,:]
        else:
            decoder_input=output_image
        for i in range(output_length):
            hidden,output_image = model(decoder_input, hidden=hidden)
            target=yy[:,i,:,:,:]
            loss+=loss_function(output_image,target)
            if use_teacher_force:
                decoder_input=target
            else:
                decoder_input=output_image
        #print(loss.item())
        train_mse.append(loss.item() / yy.shape[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mse = round((np.mean(train_mse)), 5)
    return train_mse
def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds=[]
    trues=[]
    with torch.no_grad():
        for xx, yy in valid_loader: #batch*times*c*h*w
            loss = 0
            ims=[]
            xx = xx.to(device)
            yy = yy.to(device)
            input_length = xx.size()[1]
            target_length = yy.size()[1]
            hidden=None
            for i in range(input_length-1):
                hidden,output_image=model(xx[:,i],i==0,hidden=hidden)

            decoder_input=xx[:,-1,:,:,:]
            for i in range(target_length):
                hidden,output_image=model(decoder_input, False, False, hidden=hidden)
                ims.append(output_image.cpu().data.numpy())
                decoder_input=output_image
                loss+=loss_function(output_image,yy[:,i,:,:,:])
            ims = np.array(ims).transpose(1, 0, 2, 3, 4)
           # preds.append(ims)
           # trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
       # preds=np.concatenate(preds,axis=0)
       # trues = np.concatenate(trues, axis=0)
        valid_mse=round((np.mean(valid_mse)), 8)
    return valid_mse,preds,trues

def test_epoch(valid_loader, model, loss_function):
    loss_curve = []
    preds=[]
    trues=[]
    with torch.no_grad():
        for xx, yy in valid_loader: #batch*times*c*h*w

            ims=[]
            xx = xx.to(device)
            yy = yy.to(device)
            input_length = xx.size()[1]
            target_length = yy.size()[1]
            hidden=None
            for i in range(input_length-1):
                hidden,output_image  = model(xx[:,i,:,:,:], (i==0),hidden=hidden)
            decoder_input=xx[:,-1,:,:,:]
            for i in range(target_length):
                hidden, output_image = model(decoder_input, hidden=hidden)
                ims.append(output_image.cpu().data.numpy())
                decoder_input=output_image
                loss=loss_function(output_image,yy[:,i,:,:,:])
                loss_curve.append(loss.item())
            ims = np.array(ims).transpose(1, 0, 2, 3, 4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())

        preds=np.concatenate(preds,axis=0)
        trues = np.concatenate(trues, axis=0)
        prediction_length=preds.shape[1]
        loss_curve=np.array(loss_curve).reshape(-1, prediction_length)
        loss_curve=(np.mean(loss_curve,axis=0))
    return preds,trues,loss_curve
