import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
from torch.utils import data

# import itertools
# import re
# import random
# import time
# from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
import bp3d

warnings.filterwarnings("ignore")


class IdealizedGrasslands(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc
        self.list_IDs = indices
        self.c = bp3d.Client(url='https://burnpro3d.sdsc.edu/api')
        self.ens = self.c.load_ensemble('./data/uniform-pgml-success.bp3d.json')
        self.out = self.ens.output()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        '''
        fuel=torch.np.array(self.out[ID].zarr['fuel_dens'])
        fuel=torch.FloatTensor(fuel[81:])
        y_fuel=fuel[self.mid:(self.mid + self.output_length)
        x_fuel=fuel[(self.mid - self.input_length):self.mid]  
        windu=torch.np.array(self.out[ID].zarr['windu'])
        windu=torch.FloatTensor(windu[81:])
        y_windu=windu[self.mid:(self.mid + self.output_length)
        x_windu=windu[(self.mid - self.input_length):self.mid]  
        windv=torch.np.array(self.out[ID].zarr['windv'])
        windv=torch.FloatTensor(windv[81:])
        y_windv=windv[self.mid:(self.mid + self.output_length)
        x_windv=windv[(self.mid - self.input_length):self.mid]  
        '''

        y_fuel = torch.load(self.direc + "fuel_sample" + str(ID) + ".pt")[self.mid:(self.mid + self.output_length)]
        x_fuel = torch.load(self.direc + "fuel_sample" + str(ID) + ".pt")[(self.mid - self.input_length):self.mid]
        x_windu = torch.load(self.direc + "windu_sample" + str(ID) + ".pt")[(self.mid - self.input_length):self.mid]
        x_windv = torch.load(self.direc + "windv_sample" + str(ID) + ".pt")[(self.mid - self.input_length):self.mid]
        # x=torch.stack((x_fuel,x_windu,x_windv),dim=0)
        y_windu = torch.load(self.direc + "windu_sample" + str(ID) + ".pt")[self.mid:(self.mid + self.output_length)]
        y_windv = torch.load(self.direc + "windv_sample" + str(ID) + ".pt")[self.mid:(self.mid + self.output_length)]

        # y=torch.stack((y_fuel,y_windu,y_windv),dim=0)

        return x_fuel.float(), x_windu.float(), x_windv.float(), y_fuel.float(), y_windu.float(), y_windv.float()
