# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Bi_GRU_ATT(nn.Module):

    def __init__(self):

        super(Bi_GRU_ATT, self).__init__()
        self.gru_input_dim = 6
        self.gru_hidden_dim_1 = 128
        self.hours_length = 168
        self.output_out_dim = 1
        self.dropout_rate = 0.5
        self.output_in_dim = self.gru_hidden_dim_1*3


        self.rnn = nn.GRU(self.gru_input_dim,self.gru_hidden_dim_1,batch_first=True,bidirectional=True,dropout=0.5)
        self.output = nn.Sequential(
                                    nn.Linear(self.output_in_dim, 128),
                                    nn.PReLU(),
                                    nn.Linear(128, 32),
                                    nn.PReLU(),
                                    nn.Linear(32, self.output_out_dim),
                                    nn.PReLU())


        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.linear = nn.Linear(self.gru_hidden_dim_1, 1)
        self.W = nn.Parameter(torch.nn.init.normal_(torch.Tensor(self.gru_hidden_dim_1, 1), std=0.1).cuda())



    def forward(self,s_inp,flow_x):        #s_inp [128, 168, 3]
        #Bidirectional-GRU
        s_inp = torch.cat([s_inp, flow_x], dim=2)
        z3,h3 = self.rnn(s_inp)              #torch.Size([64, 100, 256]) torch.Size([1, 64, 256])
        H_rnn = torch.cat([h3[0],h3[1]], dim=1)

        #attetion
        f_z3,b_z3 = z3.split(self.gru_hidden_dim_1,2)
        Z = f_z3+b_z3       # (128,168,128)
        M = self.tanh(Z)       #shape (128,168,128)

        self.alpha = self.softmax(M.view(-1, self.gru_hidden_dim_1).matmul(self.W).view(-1, self.hours_length))
        R = Z.permute(0,2,1).matmul(self.alpha.view([-1, self.hours_length, 1])).squeeze()
        H_att = self.prelu2(R)

        H_out = torch.cat([H_rnn, H_att], dim=1)

        #Output Layer
        pred = self.output(H_out)
        return pred




