from __future__ import print_function
import  numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import warnings; warnings.filterwarnings(action='once')
import time
import torch.nn as nn
from torch.distributions.normal import Normal

def string2timestamp(strings, T=24):
    timestamps = []
    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))
    return timestamps

def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    #print(timestamps)
    vec = [time.strptime(str(t[:8]), '%Y%m%d').tm_wday for t in timestamps]  # python3
    #print(vec)
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)

        #print(v)
    return np.asarray(ret)

def load_weekend(day_feature):
    weekend_feature = []
    for i in day_feature:
        if i >= 6:
            weekend_feature.append(1)
        else:
            weekend_feature.append(0)
    return weekend_feature

def load_hour(timestamps_all):
    hour  = []
    t_hour = 0.
    count = 0
    for i in timestamps_all:
        hour.append(t_hour)
        if count == 24-1:
            count = 0
            t_hour = 0.
        else:
            t_hour += 1
            count += 1
    return hour

def load_monthday(timestamps_all):
    month  = []
    day = []
    for i in timestamps_all:
        month.append(int(i[4:6]))
        day.append(int(i[6:8]))
    return month,day



def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))


def get_data_generator(opt,data,flow_data,timestamps,temperature_prices,T = 24,meta_data = True):
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # E data
    dataX, dataY = [], []
    flow_dataX, flow_dataY = [], []
    timestampX ,timestampY = [],[]
    tpX, tpY = [], []
    for i in range(len(data) - T*7):  # T*7
        dataX.append(data[i:(i + T*7)])
        dataY.append(data[i + T*7])

        #timestamps
        timestampX.append(timestamps[i:(i + T*7)])
        timestampY.append(timestamps[i + T*7])

        #flow  data
        flow_dataX.append(flow_data[i:(i + T*7)])
        flow_dataY.append(flow_data[i + T * 7])

        #temperature and prices
        tpX.append(temperature_prices[i:(i + T*7)])
        tpY.append(temperature_prices[i + T * 7])


    external_dim = False
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestampY)  # [[0 0 0 0 0 1 0 0],[1 0 0 0 0 0 0 1]] 第8位0表示周末，1表示工作日

        #print(time_feature)
        day_feature = [np.argmax(one_hot) + 1 for one_hot in time_feature[:, :7]]
        day_feature = np.array(day_feature)[:, np.newaxis]
        print("day: ", day_feature.min(), day_feature.max(), day_feature.shape)

        #load_monthday
        month,day = load_monthday(timestampY)
        month = np.array(month)[:, np.newaxis]
        day = np.array(day)[:, np.newaxis]
        print("month,day: ", month.min(), month.max(), month.shape,day.min(),day.max(),day.shape)

        weekend_feature = load_weekend(day_feature)
        weekend_feature = np.array(weekend_feature)[:, np.newaxis]
        print("weekend: ", weekend_feature.min(), weekend_feature.max(), weekend_feature.shape)

        hour_feature = load_hour(timestampY)
        hour_feature = np.array(hour_feature)[:, np.newaxis]

        print("hour: ", hour_feature.min(), hour_feature.max(), hour_feature.shape)

        meta_feature = np.hstack([day_feature, hour_feature, weekend_feature,month,day])
        print("meta shape", meta_feature.shape)
        external_dim = True

    #E data
    train_x = Tensor(np.array(dataX[opt.len_test:]))#[:,:,0].unsqueeze(2)
    test_x = Tensor(np.array(dataX[:opt.len_test]))#[:,:,0].unsqueeze(2)
    train_y = Tensor(np.array(dataY[opt.len_test:]).reshape([-1,3]))[:,0].unsqueeze(1)
    test_y = Tensor(np.array(dataY[:opt.len_test]).reshape([-1,3]))[:,0].unsqueeze(1)

    #flow  data
    flow_train_x = Tensor(np.array(flow_dataX[opt.len_test:]))#.unsqueeze(2)
    flow_test_x = Tensor(np.array(flow_dataX[:opt.len_test]))#.unsqueeze(2)
    flow_train_y = Tensor(np.array(flow_dataY[opt.len_test:]).reshape([-1,3]))#.unsqueeze(1)
    flow_test_y = Tensor(np.array(flow_dataY[:opt.len_test]).reshape([-1,3]))#.unsqueeze(1)

    #temperature_prices
    # tp_train_x = Tensor(np.array(tpX[opt.len_test:]))#.unsqueeze(2)
    # tp_test_x = Tensor(np.array(tpX[:opt.len_test]))#.unsqueeze(2)
    tp_train_y = Tensor(np.array(tpY[opt.len_test:]).reshape([-1,1]))#.unsqueeze(1)
    tp_test_y = Tensor(np.array(tpY[:opt.len_test]).reshape([-1,1]))#.unsqueeze(1)

    #timestamp_x
    train_meta_feature = None
    test_meta_feature = None
    if meta_data:
        train_meta_feature = Tensor(np.array(meta_feature[:-opt.len_test]))#.unsqueeze(1)
        test_meta_feature = Tensor(np.array(meta_feature[-opt.len_test:]))#.unsqueeze(1)
        print(train_x.shape,train_y.shape,test_x.shape,test_y.shape,
              flow_train_x.shape,flow_train_y.shape,flow_test_x.shape,flow_test_y.shape,
              train_meta_feature.shape,test_meta_feature.shape)


    train_data = torch.utils.data.TensorDataset(train_x, train_y, flow_train_x,flow_train_y,train_meta_feature,tp_train_y)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)

    test_data = torch.utils.data.TensorDataset(test_x,test_y,flow_test_x,flow_test_y,test_meta_feature,tp_test_y)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

    return train_dataloader, test_dataloader ,external_dim

def init_network_weights(net, std = 0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def reparameterize(mu, var):
    """
    Samples z  using the reparameterization trick.
    """
    d = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
    r = d.sample(mu.size()).squeeze(-1)
    return r * var.float() + mu.float()

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2

    if len(data.size()) == 3:
        res = data[:,:,:last_dim], data[:,:,last_dim:]

    if len(data.size()) == 2:
        res = data[:,:last_dim], data[:,last_dim:]
    return res


def linspace_vector(start, end, n_points):
    size = np.prod(start.size())
    assert(start.size() == end.size())
    if size == 1:
        res = torch.linspace(start, end, n_points)
    else:
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                torch.linspace(start[i], end[i], n_points)),0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:
        device = tensor.get_device()
    return device