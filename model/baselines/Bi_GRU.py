# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
class Bi_GRU(nn.Module):

    def __init__(self):

        super(Bi_GRU, self).__init__()
        self.flow_history = True
        self.gru_input_dim = 3
        self.gru_hidden_dim_1 = 128
        self.hours_length = 168
        self.output_out_dim = 1
        self.num_ode_blocks = 1
        self.dropout_rate = 0.5
        self.output_in_dim = self.gru_hidden_dim_1*2

        if self.flow_history:
            print("------using historical flow------")
            self.gru_input_dim += 3

        self.rnn = nn.GRU(self.gru_input_dim,self.gru_hidden_dim_1,batch_first=True,bidirectional=True,dropout=0.5)
        self.output = nn.Sequential(
                                    nn.Linear(self.output_in_dim, 128),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(128, 32),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(32, self.output_out_dim),
                                    nn.PReLU())

    def forward(self,s_inp,flow_x):        #s_inp [128, 168, 3]
        #Bidirectional-GRU
        if self.flow_history:
            s_inp = torch.cat([s_inp,flow_x], dim =2)
        z3,h3 = self.rnn(s_inp)              #torch.Size([64, 100, 256]) torch.Size([1, 64, 256])
        con = torch.cat([h3[0],h3[1]], dim=1)
        #Output Layer
        pred = self.output(con)
        return pred




