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
    for x_fuel,x_windu,x_windv, y_fuel,y_windu,y_windv in train_loader:
        input_length=x_fuel.size(1)
        output_length=y_fuel.size(1)
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
        hidden=None
        for i in range(0,input_length-1):
            xx=torch.cat((x_fuel[:,i].unsqueeze(1),x_windu[:,i].unsqueeze(1),x_windv[:,i].unsqueeze(1)),dim=1)
            hidden,output_image=model(xx,i==0,hidden=hidden)

        use_teacher_force=True if np.random.random()<teacher_force_ratio else False
        if(use_teacher_force):                     
            decoder_input=torch.cat((x_fuel[:,-1,:,:,].unsqueeze(1),x_windu[:,-1].unsqueeze(1),
                                                        x_windv[:,-1].unsqueeze(1)),
                                                          dim=1)
        else:
            decoder_input=torch.cat((output_image,x_windu[:,-1].unsqueeze(1),
                                                        x_windv[:,-1].unsqueeze(1)),
                                                          dim=1)
        for i in range(output_length):
            hidden,output_image = model(decoder_input, hidden=hidden)
            target=y_fuel[:,i,:,:].unsqueeze(1)
            loss+=loss_function(output_image,target)
            #if torch.isnan(loss):
             #   print(output_image)
            if use_teacher_force:
                decoder_input=torch.cat((target,y_windu[:,i].unsqueeze(1),
                                                        y_windv[:,i].unsqueeze(1)),
                                                          dim=1)
            else:
                decoder_input=torch.cat((output_image,y_windu[:,i].unsqueeze(1),
                                                        y_windv[:,i].unsqueeze(1)),
                                                          dim=1)
        train_mse.append(loss.item() / y_fuel.shape[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
    
    train_mse = round((np.mean(train_mse)), 5)
    return train_mse
def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds=[]
    trues=[]
    with torch.no_grad():
        for x_fuel,x_windu,x_windv, y_fuel,y_windu,y_windv in valid_loader:
            input_length=x_fuel.size(1)
            output_length=y_fuel.size(1)
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
            hidden=None
            for i in range(0,input_length-1):
                xx=torch.cat((x_fuel[:,i].unsqueeze(1),x_windu[:,i].unsqueeze(1),          
                              x_windv[:,i].unsqueeze(1)),dim=1)
                hidden,output_image=model(xx,i==0,hidden=hidden)

           
    
            decoder_input=torch.cat((output_image,x_windu[:,-1].unsqueeze(1),
                                                        x_windv[:,-1].unsqueeze(1)),
                                                          dim=1)
            for i in range(output_length):
                hidden,output_image = model(decoder_input, hidden=hidden)
                target=y_fuel[:,i,:,:].unsqueeze(1)
                loss+=loss_function(output_image,target)
    
                decoder_input=torch.cat((output_image,y_windu[:,i].unsqueeze(1),
                                                        y_windv[:,i].unsqueeze(1)),
                                                          dim=1)
  
        valid_mse.append(loss.item()/y_fuel.shape[1])

    valid_mse=round((np.mean(valid_mse)), 8)
    return valid_mse,preds,trues

def test_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds=[]
    trues=[]
    with torch.no_grad():
        loss_curve = []
        for x_fuel,x_windu,x_windv, y_fuel,y_windu,y_windv in valid_loader:
            input_length=x_fuel.size(1)
            output_length=y_fuel.size(1)
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
            hidden=None
            for i in range(0,input_length-1):
                xx=torch.cat((x_fuel[:,i].unsqueeze(1),x_windu[:,i].unsqueeze(1),          
                              x_windv[:,i].unsqueeze(1)),dim=1)
                hidden,output_image=model(xx,i==0,hidden=hidden)

           
    
            decoder_input=torch.cat((output_image,x_windu[:,-1].unsqueeze(1),
                                                        x_windv[:,-1].unsqueeze(1)),
                                                          dim=1)
            for i in range(output_length):
                hidden,output_image = model(decoder_input, hidden=hidden)
                target=y_fuel[:,i,:,:].unsqueeze(1)
                mse=loss_function(output_image,target)
                loss+=mse
                loss_curve.append(mse.item())
                ims.append(output_image.cpu().data.numpy())
                decoder_input=torch.cat((output_image,y_windu[:,i].unsqueeze(1),
                                                        y_windv[:,i].unsqueeze(1)),dim=1)
            
            ims = np.array(ims).transpose(1, 0, 2, 3, 4)
            preds.append(ims)
            trues.append(y_fuel.cpu().data.numpy())
           
            valid_mse.append(loss.item()/y_fuel.shape[1])



        preds=np.concatenate(preds,axis=0)
        trues = np.concatenate(trues, axis=0)
        prediction_length=preds.shape[1]
        loss_curve=np.array(loss_curve).reshape(-1, prediction_length)
        loss_curve=(np.mean(loss_curve,axis=0))
    return preds,trues,loss_curve
