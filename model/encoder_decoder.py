import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, gru_in_dim = 3, gru_hidden_dim = 128,z0_dim = 50, fc_dim = 128):
        super(Encoder, self).__init__()

        self.fc_dim = fc_dim
        self.gru_in_dim = gru_in_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.z0_hidden = z0_dim
        self.gru = nn.GRU(self.gru_in_dim,self.gru_hidden_dim,batch_first=True,bidirectional=True,dropout=0.5)
        self.q_z_mean = nn.Linear(self.gru_hidden_dim*2, self.z0_hidden)
        self.q_z_var = nn.Sequential(
            nn.Linear(self.gru_hidden_dim*2, self.z0_hidden),          #(256,50)
            nn.Softplus(),nn.Hardtanh(min_val=0.01, max_val=7.))


    def forward(self,input):
        z, h = self.gru(input)
        #print(z.shape,h.shape)          torch.Size([128, 168, 256]) torch.Size([2, 128, 128])
        con_h = torch.cat([h[0], h[1]], dim=1)   #shape [128,256]

        z0_mean = self.q_z_mean(con_h)
        z0_std = self.q_z_var(con_h)
        return z0_mean,z0_std,con_h

class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim=128):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.gru = nn.GRU(in_dim,hid_dim,batch_first=True,dropout=0.5)
    def forward(self, input, hidden):
        hidden = hidden.view([1,-1,self.hid_dim])
        #print(hidden.shape)
        output, hidden = self.gru(input, hidden)
        hidden = hidden.view([-1,self.hid_dim])
        return hidden

class FcDecoder(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(FcDecoder, self).__init__()
        self.hid_dim = out_dim
        self.FC = nn.Sequential(
            nn.Linear(self.hid_dim, 64),          #(256,128)
            nn.ELU(inplace=True),
            nn.Linear(64, 3), )           #(128,100)

    def forward(self, input):
        pred = self.FC(input)
        return pred