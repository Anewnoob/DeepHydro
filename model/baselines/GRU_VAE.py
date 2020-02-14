# -*- coding: utf-8 -*-
from utils.ODE import *
from model.encoder_decoder import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence,Independent



class GRU_VAE(nn.Module):

    def __init__(self):

        super(GRU_VAE, self).__init__()
        self.vae = True
        self.batch_size = 128
        self.gru_hidden_dim_1 =128
        self.gru_hidden_dim_2 = 128
        self.gru_input_dim = 6
        self.z0_hidden = 50
        self.output_out_dim = 1
        self.dropout_rate = 0.5
        self.output_in_dim = self.gru_hidden_dim_1

        #encoder-decoder
        if self.vae:
            self.encoder = Encoder(gru_in_dim = self.gru_input_dim, gru_hidden_dim = self.gru_hidden_dim_1,z0_dim = 50, fc_dim = 128)
            self.decoder = Decoder(self.gru_input_dim,self.gru_hidden_dim_2)
            self.linear = nn.Linear(self.z0_hidden, self.gru_hidden_dim_2)


        #output layer
        self.output = nn.Sequential(
                                    nn.Linear(self.output_in_dim, 128),
                                    nn.PReLU(),
                                    nn.Linear(128, 32),
                                    nn.PReLU(),
                                    nn.Linear(32, self.output_out_dim),
                                    nn.PReLU())

        self.tanh = nn.Tanh()

        ode_blocks1 = []
        for _ in range(1):
            ode_blocks1.append(ODEBlock(ODEfunc(self.gru_hidden_dim_2)))
        self.ode_blocks = nn.Sequential(*ode_blocks1)

    def forward(self,s_inp,flow_x):        #s_inp [128, 168, 3]

        batch_size, hours_length, n_dims = s_inp.size()
        s_inp = torch.cat([s_inp, flow_x], dim=2)

        #Bi-rnn encoder
        z0_mean,z0_std,_ = self.encoder(s_inp)

        #Sample z_0
        sample = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda()).sample(z0_mean.size()).squeeze(-1)
        z0 = sample * z0_std.float() + z0_mean.float()        #(128,50)

        #kl loss
        z0_distr = Normal(z0_mean, z0_std)
        z0_prior = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
        kldiv_z0 = kl_divergence(z0_distr, z0_prior)

        #decoder
        z0_fc = self.linear(z0)
        z0_in  = self.tanh(z0_fc)
        s_res,hidden = self.decoder(s_inp,z0_in)

        #prediction layer (ODE + MLP)
        output_in = self.ode_blocks(z0_in)
        pred = self.output(output_in)

        gaussian = Independent(Normal(loc=pred, scale=0.01), 1)
        return pred,gaussian,kldiv_z0,s_res


