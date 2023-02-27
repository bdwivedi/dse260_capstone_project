import sys

sys.path.append('../..')
import bp3d
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import torch

c = bp3d.Client(url='https://burnpro3d.sdsc.edu/api')
ens = c.load_ensemble('uniform-pgml-success.bp3d.json')
out = ens.output()
fuel_array = []
windu_array = []
windv_array = []
windw_array = []
moisture_array = []


for i in range(650):
    print(i)
    fuel=np.array(out[i].zarr['fuels-dens'])
    torch.save(torch.FloatTensor(fuel[81:]).clone(),"fuel_sample" + str(i) + ".pt")
    windu=(np.array(out[i].zarr['windu']))
    windv=(np.array(out[i].zarr['windv']))
    torch.save(torch.FloatTensor(windv[81:]).clone(),"windv_sample" + str(i) + ".pt")
    torch.save(torch.FloatTensor(windu[81:]).clone(),"windu_sample" + str(i) + ".pt")
    #np.save("/Users/oaa/Downloads/WIFIRE-master/data/wind/windu" + str(i) + ".npy",windu)
    #np.save("/Users/oaa/Downloads/WIFIRE-master/data/wind/windv" + str(i) + ".npy",windv)
    # windu_array.append(np.array(out[i].zarr['windu']))
    # windv_array.append(np.array(out[i].zarr['windv']))
    # windw_array.append(np.array(out[i].zarr['windw']))
   # moisture = np.load("/root/autodl-tmp/datas/moisture_sample" + str(i) + ".npy")
    #torch.save(torch.FloatTensor(moisture[81:]).clone(),
        #       "/root/autodl-tmp/data/moisture_sample" + str(i) + ".pt")

'''
c=bp3d.Client(url='https://burnpro3d.sdsc.edu/api')
ens = c.load_ensemble('uniform-pgml-success.bp3d.json')
out = ens.output()
fuel_array=[]
windu_array=[]
windv_array=[]
windw_array=[]
moisture_array=[]
for i in range(10):
    #print(i)
    fuel_array.append(np.array(out[i].zarr['fuels-dens']))

   # windu_array.append(np.array(out[i].zarr['windu']))
   # windv_array.append(np.array(out[i].zarr['windv']))
   # windw_array.append(np.array(out[i].zarr['windw']))
    moisture_array.append(np.array(out[i].zarr['fuels-moisture']))
fuel=np.stack(fuel_array,axis=0)
moisture=np.stack(moisture_array,axis=0)
#windu=np.stack(windu_array,axis=0)
#windv=np.stack(windv_array,axis=0)
#windw=np.stack(windw_array,axis=0)
np.save("fuel.txt",fuel)
np.save("moisture.txt,moisture)
#np.save("windu.txt",windu)
#np.save("windv.txt",windu)
#np.save("windw.txt",windw)

fuel=np.load("fuel.npy")
moisture=np.load("moisture.npy)
#windu=np.load("/home/featurize/data/windu.npy")
#windv=np.load("/home/featurize/data/windv.npy")
#windw=np.load("windw.npy")
fuel=torch.tensor(fuel)
moisture=torch.tensor(moisture)
#windv=torch.tensor(windv)
#windu=torch.tensor(windu)
#windw=torch.tensor(windw)
'''
'''
#moisture=np.load("/root/autodl-tmp/moisture.npy")
#moisture=torch.tensor(moisture)
for i in range(moisture.shape[0]):
    #print(i)
    for j in range(550):
        #torch.save(fuel[i,j:j+50].clone(),"/root/autodl-tmp/datas/fuel_sample"+str(i*550+j)+".pt")
         #torch.save(moisture[i,j:j+50].clone(),"/root/autodl-tmp/datas/moisture_sample"+str(i*550+j)+".pt")        
        torch.save(torch.FloatTensor(windv[i, j:j +31,0]).clone(), "/home/featurize/data/datas/windv_sample" + str(i * 550 + j) + ".pt")
        torch.save(torch.FloatTensor(windu[i, j:j + 31,0]).clone(), "/home/featurize/data/datas/windu_sample" + str(i * 550 + j) + ".pt")
        #torch.save(torch.FloatTensor(windw[i, j:j + 50]).clone(), "/datas/windw_sample" + str(i * 550 + j) + ".pt")
'''