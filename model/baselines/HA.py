# -*- coding: utf-8 -*-
from utils.metrics import get_MAE,get_MAPE,get_MSE
from utils.helper import *
import pickle
import os
DATAPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
year = 2017
# Load Data
preprocess_file = DATAPATH+ f'/data/PDS_{year}_data.pkl'
print(preprocess_file)
# DGS_data = {"data": DGS_DG_norm, "LL_data": DGS_flow_norm, "temp": temperature, "date": DGS_DG_T,
#             "E_min": mmn._min, "E_max": mmn._max, "LL_min": ll_mmn._min, "LL_max": ll_mmn._max}
fpkl = open(preprocess_file, 'rb')
data = pickle.load(fpkl)
fpkl.close()
DGS_DG_norm = data['data']
mmn = data['E_mmn']

look_back = 7
dataX, dataY = [], []
for i in range(len(DGS_DG_norm) - look_back):
    x = DGS_DG_norm[i:(i + look_back)]
    y = DGS_DG_norm[i + look_back]
    x = np.array(x)
    y = np.array(y)
    dataX.append(x)
    dataY.append(y)

train = dataX
test = dataY

total_mse = 0
total_mae = 0
total_mape = 0
for i in range(len(train)):
    pred_y = mmn.inverse_transform(np.mean(train[i]))
    y = np.array(mmn.inverse_transform(test[i])[0])
    total_mse += get_MSE(pred_y, y)
    total_mae += get_MAE(pred_y, y)
    total_mape += get_MAPE(pred_y, y)
rmse = np.sqrt(total_mse / len(train))
mae = total_mae / len(train)
mape = total_mape / len(train)
print (rmse,mae,mape)