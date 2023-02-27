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

__all__ = ['M2K', 'K2M']
constraints=torch.zeros((49,7,7)).to(device)
ind=0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j]=1
        ind+=1

def _apply_axis_left_dot(x, mats):
    assert x.dim() == len(mats) + 1
    sizex = x.size()
    k = x.dim() - 1
    for i in range(k):
        x = tensordot(mats[k - i - 1], x, dim=[1, k])
    x = x.permute([k, ] + list(range(k))).contiguous()
    x = x.view(sizex)
    return x


def _apply_axis_right_dot(x, mats):
    assert x.dim() == len(mats) + 1
    sizex = x.size()
    k = x.dim() - 1
    x = x.permute(list(range(1, k + 1)) + [0, ])
    for i in range(k):
        x = tensordot(x, mats[i], dim=[0, 0])
    x = x.contiguous()
    x = x.view(sizex)
    return x


class _MK(nn.Module):
    def __init__(self, shape):
        super(_MK, self).__init__()
        self._size = torch.Size(shape)
        self._dim = len(shape)
        M = []
        invM = []
        assert len(shape) > 0
        j = 0
        for l in shape:
            M.append(zeros((l, l)))
            for i in range(l):
                M[-1][i] = ((arange(l) - (l - 1) // 2) ** i) / factorial(i)
            invM.append(inv(M[-1]))
            self.register_buffer('_M' + str(j), torch.from_numpy(M[-1]))
            self.register_buffer('_invM' + str(j), torch.from_numpy(invM[-1]))
            j += 1

    @property
    def M(self):
        return list(self._buffers['_M' + str(j)] for j in range(self.dim()))

    @property
    def invM(self):
        return list(self._buffers['_invM' + str(j)] for j in range(self.dim()))

    def size(self):
        return self._size

    def dim(self):
        return self._dim

    def _packdim(self, x):
        assert x.dim() >= self.dim()
        if x.dim() == self.dim():
            x = x[newaxis, :]
        x = x.contiguous()
        x = x.view([-1, ] + list(x.size()[-self.dim():]))
        return x

    def forward(self):
        pass


class M2K(_MK):
    """
    convert moment matrix to convolution kernel
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        m2k = M2K([5,5])
        m = torch.randn(5,5,dtype=torch.float64)
        k = m2k(m)
    """

    def __init__(self, shape):
        super(M2K, self).__init__(shape)

    def forward(self, m):
        """
        m (Tensor): torch.size=[...,*self.shape]
        """
        sizem = m.size()
        m = self._packdim(m)
        m = _apply_axis_left_dot(m, self.invM)
        m = m.view(sizem)
        return m


class K2M(_MK):
    """
    convert convolution kernel to moment matrix
    Arguments:
        shape (tuple of int): kernel shape
    Usage:
        k2m = K2M([5,5])
        k = torch.randn(5,5,dtype=torch.float64)
        m = k2m(k)
    """

    def __init__(self, shape):
        super(K2M, self).__init__(shape)

    def forward(self, k):
        """
        k (Tensor): torch.size=[...,*self.shape]
        """
        sizek = k.size()
        k = self._packdim(k)
        k = _apply_axis_left_dot(k, self.M)
        k = k.view(sizek)
        return k


def tensordot(a, b, dim):
    """
    tensordot in PyTorch, see numpy.tensordot?
    """
    l = lambda x, y: x * y
    if isinstance(dim, int):
        a = a.contiguous()
        b = b.contiguous()
        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-dim]
        sizea1 = sizea[-dim:]
        sizeb0 = sizeb[:dim]
        sizeb1 = sizeb[dim:]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    else:
        adims = dim[0]
        bdims = dim[1]
        adims = [adims, ] if isinstance(adims, int) else adims
        bdims = [bdims, ] if isinstance(bdims, int) else bdims
        adims_ = set(range(a.dim())).difference(set(adims))
        adims_ = list(adims_)
        adims_.sort()
        perma = adims_ + adims
        bdims_ = set(range(b.dim())).difference(set(bdims))
        bdims_ = list(bdims_)
        bdims_.sort()
        permb = bdims + bdims_
        a = a.permute(*perma).contiguous()
        b = b.permute(*permb).contiguous()

        sizea = a.size()
        sizeb = b.size()
        sizea0 = sizea[:-len(adims)]
        sizea1 = sizea[-len(adims):]
        sizeb0 = sizeb[:len(bdims)]
        sizeb1 = sizeb[len(bdims):]
        N = reduce(l, sizea1, 1)
        assert reduce(l, sizeb0, 1) == N
    a = a.view([-1, N])
    b = b.view([N, -1])
    c = a @ b
    return c.view(sizea0 + sizeb1)
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
        hidden1,hidden2=None,None
        for i in range(input_length-1):
            hidden1,hidden2,output_image,_,_=model(xx[:,i],i==0,hidden1=hidden1,hidden2=hidden2)

        use_teacher_force=True if np.random.random()<teacher_force_ratio else False
        if(use_teacher_force):
            decoder_input=xx[:,-1,:,:,:]
        else:
            decoder_input=output_image
        for i in range(output_length):
            hidden1, hidden2, output_image, _, _ = model(decoder_input, hidden1=hidden1, hidden2=hidden2)
            target=yy[:,i,:,:,:]
            loss+=loss_function(output_image,target)
            if use_teacher_force:
                decoder_input=target
            else:
                decoder_input=output_image
        #print(loss)
        k2m=K2M([7,7]).to(device)
        for b in range(0, model.phycell.cell_list[0].input_dim):
            filters = model.phycell.cell_list[0].F.conv1.weight[:, b, :, :]  # (nb_filters,7,7)
            m = k2m(filters.double())
            m = m.float()
            loss+=loss_function(m,constraints)
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
            hidden1,hidden2 = None, None
            for i in range(input_length-1):
                hidden1,hidden2, _,_,_  = model(xx[:,i,:,:,:], (i==0),
                                                                hidden1=hidden1,hidden2=hidden2)
            decoder_input=xx[:,-1,:,:,:]
            for i in range(target_length):
                hidden1,hidden2,output_image,_,_=model(decoder_input, False, False,
                                                                                 hidden1=hidden1,hidden2=hidden2)
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
            hidden1,hidden2 = None, None
            for i in range(input_length-1):
                hidden1,hidden2, _,_,_  = model(xx[:,i,:,:,:], (i==0),
                                                                hidden1=hidden1,hidden2=hidden2)
            decoder_input=xx[:,-1,:,:,:]
            for i in range(target_length):
                hidden1,hidden2,output_image,_,_=model(decoder_input, False, False,
                                                                                 hidden1=hidden1,hidden2=hidden2)
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
