# -*- coding: utf-8 -*-
from __future__ import print_function
from utils.ODE import *
from model.encoder_decoder import *
from torch.distributions.normal import Normal
from torch.distributions import Independent
from utils.helper import reparameterize

class Planar(nn.Module):
    def __init__(self):
        super(Planar, self).__init__()
        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        return 1 - self.h(x)**2

    def forward(self, zk, u, w, b):
        zk = zk.unsqueeze(2)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w**2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        #logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        return z, log_det_jacobian


#zk
def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm

#z0
def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm

class PlanarVAE(nn.Module):

    def __init__(self):

        super(PlanarVAE, self).__init__()
        self.vae = True
        self.batch_size = 128
        self.gru_hidden_dim_1 =384
        self.gru_hidden_dim_2 = 512
        self.gru_input_dim = 6
        self.z0_hidden = 128
        self.output_out_dim = 1
        self.dropout_rate = 0.5
        self.output_in_dim =512
        #encoder-decoder
        if self.vae:
            self.encoder = Encoder(gru_in_dim = self.gru_input_dim, gru_hidden_dim = self.gru_hidden_dim_1,z0_dim = self.z0_hidden, fc_dim = 128)
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
            ode_blocks1.append(ODEBlock(ODEfunc(self.gru_hidden_dim_2),method='dopri5',rtol=1e-5,atol=1e-5))
        self.ode_blocks = nn.Sequential(*ode_blocks1)

        # Flow parameters
        flow = Planar
        self.num_flows = 6

        # Amortized flow parameters
        self.amor_u = nn.Linear(768, self.num_flows * self.z0_hidden)
        self.amor_w = nn.Linear(768, self.num_flows * self.z0_hidden)
        self.amor_b = nn.Linear(768, self.num_flows)

        # Normalizing flow layers
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)



    def forward(self,s_inp,flow_x):        #s_inp [128, 168, 3]
        batch_size, hours_length, n_dims = s_inp.size()

        s_inp = torch.cat([s_inp,flow_x], dim =2)

        #Bi-rnn encoder
        z0_mean,z0_std,con_h = self.encoder(s_inp)

        # return amortized u an w for all flows
        u = self.amor_u(con_h).view(batch_size, self.num_flows, self.z0_hidden, 1)
        w = self.amor_w(con_h).view(batch_size, self.num_flows, 1, self.z0_hidden)
        b = self.amor_b(con_h).view(batch_size, self.num_flows, 1, 1)

        # Sample z_0
        z = [reparameterize(z0_mean, z0_std)]

        # Normalizing flows
        sum_ldj = 0.
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            sum_ldj += torch.sum(log_det_jacobian.view(-1))

        z0 = z[0]
        zk = z[-1]

        # NF-KL
        log_p_zk = log_normal_standard(zk, dim=1)
        log_q_z0 = log_normal_diag(z0, mean=z0_mean, log_var=z0_std.log(), dim=1)
        sum_logs = torch.sum(log_q_z0 - log_p_zk)
        kl = sum_logs - sum_ldj

        #decoder
        z0_fc = self.linear(zk)
        z0_in  = self.tanh(z0_fc)
        s_res,hidden = self.decoder(s_inp,z0_in)

        #prediction layer (ODE + MLP)
        output_in = self.ode_blocks(z0_in)
        pred = self.output(output_in)
        gaussian = Independent(Normal(loc=pred, scale=0.01), 1)
        return pred,gaussian,kl,s_res