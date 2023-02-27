# WIFIRE

## Description


1. data/:

  i. You probably need to first run pre_process.py to pre_process the data.
  
  ii. dataset.py -  1 second idealized grassland dataset only for the fuel density
  
  iii. dataset_attension.py - 1 second idealized grassland dataset for both the fuel density and wind information


## Training



For training Unet model which takes fuel density as input:

```python train_unet.py ```

For testing Unet model which takes fuel density as input:

```python test_unet.py ```

For training Unet model which takes both fuel density and wind information as input:

```python train_unet_wind.py ```

For tesing Unet model which takes both fuel density and wind information as input:

```python test_unet_wind.py ```

The other model can be trained in the same way,