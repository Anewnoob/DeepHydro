# DeepHydro
code for "Forecasting the Evolution of Hydropower Generation"

# Prerequisites
Python >= 3.6

PyTorch >= 1.1.0

Numpy >= 1.15.1

Install torchdiffeq from https://github.com/rtqichen/torchdiffeq.

# Visualization
Power stations distribution of Dadu River:
![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/power-distribution/power-distribution-1.png)

Continuous latent representation learned in CL-RNN:
![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/CLRNN.jpg)
<div align=center><img src="https://github.com/Anewnoob/DeepHydro/blob/master/png/CLRNN.jpg" width="300" height="450" /></div>

 the process of transforming $\mathbf{z}$


# Dateset
We use two different types datasets, namely DGS(large-scale, 1/1/2017--31/12/2018) and PDS(small-scale, 1/1/2017--31/12/2018), to demonstrate DeepHydro performs the best against other baselines. The data of last 11 weeks (77 days) of the year are used for testing, and the rest for training. The more detailed descriptions can be obtained in the paper. 
