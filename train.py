# -*- coding: utf-8 -*-
from __future__ import print_function
from utils.metrics import *
from utils.helper import *
import argparse
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
import warnings
import pickle

DATAPATH = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 128, help='batchsize')
parser.add_argument('--delta', type=float, default=0.5, help='delta')
parser.add_argument('--epsilon', type=float,default=1e-7)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float,default=0.999)
parser.add_argument('--lr', type=float,default=1e-4)
parser.add_argument('--target_year', type=str, default='2018')
parser.add_argument('--num_epochs',type=int, default=150)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--logdir', default='train', help='log_path')
parser.add_argument('--model_path',default='./model_save', help='manual seed')
parser.add_argument('--save', type=str, default='./experiment')
parser.add_argument('--model', type=str, default='DeepHydro',choices=["LSTM","Bi_GRU", "Bi_GRU_ATT","GRU_VAE","PlanarVAE",
                                                       "LatentODE","DeepHydro"])
parser.add_argument('--CACHEDATA', type=bool, default = True)
parser.add_argument('--T', type=int, default=24)
parser.add_argument('--len_test', type=int, default=24*7*11)
parser.add_argument('--ext', type=bool, default=True)
parser.add_argument('--harved_epoch', type=int, default=30)
parser.add_argument('--dataset', type=str, default='PDS',choices=["DGS","PDS"],help='which dataset to use')
parser.add_argument('--sample_interval', type=int, default=60,help='interval between validation')
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

warnings.filterwarnings('ignore')
save_path = 'saved_model/{}/{}-{}'.format(opt.dataset,
                                             opt.model,
                                             opt.ext)
os.makedirs(save_path, exist_ok=True)

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:",device)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

iter = 0
rmses = [np.inf]
maes = [np.inf]

# Load Data
preprocess_file = DATAPATH+ f'/data/{opt.dataset}_{opt.target_year}_data.pkl'
print(preprocess_file)
# DGS_data = {"data": DGS_DG_norm, "LL_data": DGS_flow_norm, "temp_prices": TP, "date": DGS_DG_T,
#             "E_min": mmn._min, "E_max": mmn._max, "LL_min": ll_mmn._min, "LL_max": ll_mmn._max}
fpkl = open(preprocess_file, 'rb')
data = pickle.load(fpkl)
fpkl.close()
DGS_DG_norm = data['data']
DGS_flow_norm = data['LL_data']
DGS_DG_T = data['date']
mmn = data['E_mmn']
LL_mmn = data['LL_mmn']
temperature_prices = data['temp_prices']
print(f'data length:{DGS_DG_norm.shape,DGS_flow_norm.shape,DGS_DG_T.shape,temperature_prices.shape}')


# model
print("feature layer:", opt.model)
if opt.model == 'LSTM':
    from model.baselines.LSTM import LSTM
    model = LSTM()
elif opt.model == 'Bi_GRU':
    from model.baselines.Bi_GRU import Bi_GRU
    model = Bi_GRU()
elif opt.model == 'Bi_GRU_ATT':
    from model.baselines.Bi_GRU_ATT import Bi_GRU_ATT
    model = Bi_GRU_ATT()
elif opt.model == 'GRU_VAE':
    from model.baselines.GRU_VAE import GRU_VAE
    model = GRU_VAE()
elif opt.model == 'PlanarVAE':
    from model.baselines.PlanarVAE import PlanarVAE
    model = PlanarVAE()
elif opt.model == 'LatentODE':
    from model.baselines.LatentODE import LatentODE
    model = LatentODE(6, 128, n_gru_units=100,n_ode_gru_dims = 128,device = device).to(device)
else:
    from model.DeepHydro import DeepHydro
    model = DeepHydro(use_external  = opt.ext)

loss_fn = nn.MSELoss()
model.apply(init_network_weights)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e10)
if cuda:
    model.cuda()
    loss_fn.cuda()
lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
print_model_parm_nums(model, opt.model)

if __name__ == '__main__':
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    train_batch_generator,test_batch_generator,external_dim = get_data_generator(opt,DGS_DG_norm,DGS_flow_norm,DGS_DG_T,temperature_prices)
    s_time = datetime.now()

    #train
    for epoch in range(opt.num_epochs):
            ep_time = datetime.now()
            beta = min([(epoch * 1.) / max([100, 1.]), 1.])
            for i, (x_seq,y_batch,flow_x,flow_y,T,TP) in enumerate(train_batch_generator):
                model.train()
                optimizer.zero_grad()
                """LSTM,Bi-GRU,Bi-GRU-ATT"""
                if opt.model == 'LSTM' or opt.model == 'Bi_GRU' or opt.model == 'Bi_GRU_ATT':
                    x_res = model(x_seq,flow_x)
                    loss = loss_fn(x_res, y_batch)


                """GRU-VAE"""
                if opt.model == 'GRU_VAE' or opt.model == 'LatentODE':
                    x_res, gaussian, kl_divergence = model(x_seq,flow_x)
                    gaussian_likelihood = gaussian.log_prob(y_batch)
                    gaussian_likelihood = torch.mean(gaussian_likelihood)
                    kl_divergence = torch.mean(kl_divergence)
                    ELBO = gaussian_likelihood - beta  * kl_divergence
                    loss = -ELBO
                    #print('loss:',format(loss,'.5f'),'KL:',format(kl_divergence,'.8f'),beta)

                """PlanarVAE, LatentODE"""
                if opt.model == 'PlanarVAE':
                    x_res,gaussian,kl_divergence = model(x_seq,flow_x)
                    kl_divergence /= float(len(x_seq))
                    mse_loss = loss_fn(x_res,y_batch)               #(128,3)  (128,3)  [:][1]
                    gaussian_likelihood = gaussian.log_prob(y_batch)
                    gaussian_likelihood = torch.mean(gaussian_likelihood)
                    ELBO = gaussian_likelihood - beta * kl_divergence
                    loss = -ELBO
                    #print('mse_loss:',format(mse_loss,'.5f'),'loss:',format(loss,'.5f'),'KL:',format(kl_divergence,'.8f'))

                """DeepHydro"""
                if opt.model == 'DeepHydro':
                    pred_y,gaussian,kl_divergence,rec_x_seq = model(x_seq,flow_x,flow_y,T,TP)
                    mse_loss = loss_fn(pred_y,y_batch)+ loss_fn(rec_x_seq,torch.cat([x_seq,flow_x], dim =2)[:,-1,:])         #(128,3)  (128,3)  [:][1]
                    gaussian_likelihood = gaussian.log_prob(y_batch)
                    loss = mse_loss - torch.mean(gaussian_likelihood,0) + beta * kl_divergence
                    #print('loss:', loss.item(),mse_loss.item(),'kl_loss:', kl_divergence.item(), beta)

                loss.backward()
                optimizer.step()
                iter += 1

    #save model
    torch.save(model.state_dict(), '{}/{}.pt'.format(save_path, opt.model))
    #test
    total_mse, total_mae, total_mape = 0, 0, 0
    model.eval()
    valid_time = datetime.now()
    with torch.no_grad():
        for i, (x_seq, y_batch, flow_x, flow_y, T, TP) in enumerate(test_batch_generator):
            if opt.model == 'LSTM' or opt.model == 'Bi_GRU' or opt.model == 'Bi_GRU_ATT':
                x_res = model(x_seq, flow_x)

            if opt.model == 'GRU_VAE' or opt.model == 'PlanarVAE' or opt.model == 'LatentODE' :
                x_res,_,_ = model(x_seq,flow_x)

            if opt.model == 'DeepHydro':
                x_res,_,_,_ = model(x_seq,flow_x,flow_y,T,TP)

            x_res = mmn.inverse_transform(x_res.cpu().detach().numpy())
            y_batch = mmn.inverse_transform(y_batch.cpu().detach().numpy())
            total_mse += get_MSE(x_res, y_batch) * len(x_seq)
            total_mae += get_MAE(x_res, y_batch) * len(x_seq)
            total_mape += get_MAPE(x_res, y_batch) * len(x_seq)
    rmse = np.sqrt(total_mse / len(test_batch_generator.dataset))
    mae = total_mae / len(test_batch_generator.dataset)
    mape = total_mape / len(test_batch_generator.dataset)
    print('Test\tRMSE\t{:.6f}\tMAE\t{:.6f}\tMAPE\t{:.6f}'.format(rmse, mae, mape))