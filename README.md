# DeepHydro
Code for "Forecasting the Evolution of Hydropower Generation"

# Prerequisites
Python >= 3.6

PyTorch >= 1.1.0

Numpy >= 1.15.1

Install torchdiffeq from https://github.com/rtqichen/torchdiffeq.

# Visualization
The distributions of latent representation along with the process of learning temporal dependencies in our CL-RNN model:
<div align=center><img src="https://github.com/Anewnoob/DeepHydro/blob/master/png/CLRNN.jpg" width="700" height="240" /></div>

The process of transforming latent representation via continuous normalizing flow:
<div align=center><img src="https://github.com/Anewnoob/DeepHydro/blob/master/png/cnf.jpg" width="700" height="400" /></div>


# Dateset
We use two different types datasets, namely DGS(large-scale, 1/1/2017--31/12/2018) and PDS(small-scale, 1/1/2017--31/12/2018), to demonstrate DeepHydro performs the best against other baselines. The data of last 11 weeks (77 days) of the year are used for testing, and the rest for training. The more detailed descriptions can be obtained in the paper. 
Power stations distribution of Dadu River:
![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/power-distribution/power-distribution-1.png)

# Prediction
To better observe the visualization of predcition, we randomly select the data of one week on DGS dataset and plot the predicted results of DeepHydro and ground truth for comparison:
![image](https://github.com/Anewnoob/DeepHydro/blob/master/png/GT-1-week-dgs/GT-1-week-dgs-1.png)
<div align=center><img src="https://github.com/Anewnoob/DeepHydro/blob/master/png/GT-1-week-dgs/GT-1-week-dgs-1.png" width="700" height="400" /></div>
