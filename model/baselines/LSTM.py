# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
class LSTM(nn.Module):
    def __init__(self,):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(6, 128,batch_first=True)
        self.FC = nn.Sequential(
                nn.Linear(128, 64),
                nn.PReLU(),
                nn.Linear(64,1),
                nn.PReLU(),
            )

    def forward(self,s_inp,flow_x):
        s_inp = torch.cat([s_inp, flow_x], dim=2)
        x, (h,c) = self.rnn(s_inp)
        output_in_last_timestep=x[:,-1,:]
        seq_output = self.FC(output_in_last_timestep)
        return seq_output
