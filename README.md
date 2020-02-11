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
![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/CL-RNN/4th-1.png)![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/CL-RNN/16th-1.png)![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/CL-RNN/64th-1.png)![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/CL-RNN/128th-1.png)


# Dateset
We use two different types datasets, namely DGS(large-scale, 1/1/2017--31/12/2018) and PDS(small-scale, 1/1/2017--31/12/2018), to demonstrate DeepHydro performs the best against other baselines. The data of last 11 weeks (77 days) of the year are used for testing, and the rest for training. The more detailed descriptions can be obtained in the paper. 
