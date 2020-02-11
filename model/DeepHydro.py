# -*- coding: utf-8 -*-
from utils.ODE import *
from utils.helper import init_network_weights,reparameterize,split_last_dim,linspace_vector
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence,Independent
from utils.cnf import ODECNFfunc,CNF

#GRUCell -- update h and z
class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=128):
        super(GRU_unit, self).__init__()
        #self.fn = nn.Softplus()

        if update_gate is None:
            self.update_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            init_network_weights(self.update_gate)
        else:
            self.update_gate = update_gate

        if reset_gate is None:
            self.reset_gate = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim),
                nn.Sigmoid())
            init_network_weights(self.reset_gate)
        else:
            self.reset_gate = reset_gate

        if new_state_net is None:
            self.new_state_net = nn.Sequential(
                nn.Linear(latent_dim * 2 + input_dim, n_units),
                nn.Tanh(),
                nn.Linear(n_units, latent_dim * 2))
            init_network_weights(self.new_state_net)
        else:
            self.new_state_net = new_state_net

    def forward(self, y_mean, y_std, x):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert (not torch.isnan(new_y).any())
        assert (not torch.isnan(new_y_std).any())

        new_y_std = new_y_std.abs()
        return new_y, new_y_std

def run_rnn(self,inputs, cell,z_cell,ode_solver,ode_solver1,first_hidden=None):
    batch_size, n_steps, n_dims = inputs.size()
    if n_steps == 0:
        n_steps = n_steps

    all_hiddens = []
    all_z_hidden = []
    all_mean = []
    all_std = []
    hidden = first_hidden
    z = first_hidden

    prev_y = torch.zeros((batch_size, 128)).cuda()
    prev_std = torch.zeros((batch_size, 128)).cuda()
    prev_z_mean = torch.zeros((batch_size, 50)).cuda()
    prev_z_std = torch.zeros((batch_size, 50)).cuda()

    if hidden is not None:
        all_hiddens.append(hidden)
        n_steps -= 1

    for i in range(n_steps):
        rnn_input = inputs[:, i]
        hidden, hidden_std = cell(prev_y, prev_std, rnn_input)
        z, z_std = z_cell(prev_z_mean, prev_z_std,hidden)
        z = ode_solver(z)

        assert (not torch.isnan(z).any())
        # z_std = ode_solver1(z_std)
        # assert (not torch.isnan(z_std).any())
        z_hidden = reparameterize(z, z_std)

        prev_y, prev_std = hidden, hidden_std
        prev_z_mean, prev_z_std = z, z_std
        all_hiddens.append(hidden)
        all_z_hidden.append(z_hidden)
        all_mean.append(z)
        all_std.append(z_std)

    #CNF is implemented by https://github.com/rtqichen/ffjord
    if self.CNF:
        # CNF
        z0 = reparameterize(z, z_std)
        zero = torch.zeros(inputs.shape[0], 1).to(inputs)
        zk, delta_logp = self.cnf_blocks(z0, zero)
        # CNF-KL
        log_p_zk = log_normal_standard(zk, dim=1)
        log_q_z0 = log_normal_diag(z0, mean=z, log_var=z_std.log(), dim=1)
        sum_logs = torch.sum(log_q_z0 - log_p_zk)
        sum_ldj = torch.sum(-delta_logp.view(-1))
        kl_all = sum_logs - sum_ldj
        kl_all /= float(batch_size)
        z = zk# - delta_logp
    else:
        z_distr = Normal(z, z_std)
        z_prior = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
        kl_z = kl_divergence(z_distr, z_prior)
        kl_z = torch.sum(kl_z)
        kl_all = kl_z / float(batch_size)

    all_hiddens = torch.stack(all_hiddens, 0)
    all_hiddens = all_hiddens.permute(1, 0, 2)
    all_z_hidden = torch.stack(all_z_hidden, 0)
    all_z_hidden = all_z_hidden.permute(1, 0, 2)
    all_mean = torch.stack(all_mean, 0)
    all_mean = all_mean.permute(1, 0, 2)
    all_std = torch.stack(all_std, 0)
    all_std = all_std.permute(1, 0, 2)
    return hidden,all_hiddens,z,all_z_hidden,kl_all,all_mean,all_std

class DeepHydro(nn.Module):

    def __init__(self,use_external = True):

        super(DeepHydro, self).__init__()
        self.flow_history = True
        self.use_time_info = use_external
        self.use_ll_info = use_external
        self.temp_prices = False
        self.CNF = True
        self.batch_size = 128
        self.gru_hidden_dim_1 =128
        self.T_out_dim = 128
        self.LL_out_dim = 128
        self.tp_out_dim = 128
        self.hours_length = 168
        self.gru_input_dim = 3
        self.z0_hidden = 50
        self.output_out_dim = 1
        self.num_ode_blocks = 1
        self.dropout_rate = 0.5
        self.output_in_dim = self.gru_hidden_dim_1*2 + self.z0_hidden

        if self.flow_history:
            print("------using historical water flow------")
            self.gru_input_dim += 3

        #CL-RNN cell
        self.rnn_cell = GRU_unit(self.gru_hidden_dim_1, self.gru_input_dim, n_units = 128).cuda()
        self.z_cell = GRU_unit(50, 128, n_units = 100).cuda()

        #Bi-GRU -- extract global temporal feature gt
        self.rnn = nn.GRU(self.gru_hidden_dim_1+self.z0_hidden, self.gru_hidden_dim_1, batch_first=True,
                          bidirectional=True, dropout=0.5)

        ode_blocks1 = []
        for _ in range(self.num_ode_blocks):
            ode_blocks1.append(ODEBlock(ODEfunc(self.z0_hidden),method='dopri5',rtol=1e-5,atol=1e-5))
        self.decoder = nn.Sequential(*ode_blocks1)

        ode_blocks2 = []
        for _ in range(self.num_ode_blocks):
            ode_blocks2.append(ODEBlock(ODEfunc(self.z0_hidden,fn = nn.Tanh()),
                                        integration_time = torch.tensor([0,1,50]).float(),method='euler',rtol=1e-3,atol=1e-4))
        self.z_mean_ode = nn.Sequential(*ode_blocks2).cuda()
        ode_blocks3 = []
        for _ in range(self.num_ode_blocks):
            ode_blocks3.append(ODEBlock(ODEfunc(self.z0_hidden,fn = nn.Tanh()),
                                        integration_time = torch.tensor([0,1,50]).float(),method='euler',rtol=1e-3,atol=1e-4))
        self.z_std_ode = nn.Sequential(*ode_blocks3).cuda()

        #external factors extraction network (EFEN)
        #temperol information
        if self.use_time_info:
            print("------using temporal information------")
            self.output_in_dim += self.T_out_dim            #128+128
            self.embed_week = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
            self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]

            self.fcn_1 = nn.Sequential(
                nn.Linear(6, 32),
                nn.ELU(inplace=True),
                nn.Linear(32, self.T_out_dim),
            )
            ode_blocks4 = []
            for _ in range(self.num_ode_blocks):
                ode_blocks4.append(ODEBlock(ODEfunc(self.T_out_dim)))
            self.T_ode = nn.Sequential(*ode_blocks4)

        #water flow
        if self.use_ll_info:
            self.output_in_dim += self.LL_out_dim           # 256+128
            print("------using water flow factor------")
            self.fcn_2 = nn.Sequential(
                nn.Linear(3, 32),
                nn.ELU(inplace=True),
                nn.Linear(32, self.LL_out_dim),
            )

            ode_blocks5 = []
            for _ in range(self.num_ode_blocks):
                ode_blocks5.append(ODEBlock(ODEfunc(self.LL_out_dim)))
            self.LL_ode_Y = nn.Sequential(*ode_blocks5)


        self.reconstruct = nn.Sequential(
            nn.Linear(self.z0_hidden, 6),
            nn.PReLU(),
        )

        # temperature and  electricity prices
        if self.temp_prices:
            self.output_in_dim += self.tp_out_dim  # 256+128
            print("------using temperature and electricity prices factors------")
            self.fcn_2 = nn.Sequential(
                nn.Linear(2, 32),
                nn.ELU(inplace=True),
                nn.Linear(32, self.tp_out_dim),
            )

            ode_blocks6 = []
            for _ in range(self.num_ode_blocks):
                ode_blocks6.append(ODEBlock(ODEfunc(self.tp_out_dim)))
            self.temp_prices_ode = nn.Sequential(*ode_blocks6)

        self.reconstruct = nn.Sequential(
            nn.Linear(self.z0_hidden, 6),
            nn.PReLU(),
        )

        def construct_cnf():
            diffeq = ODECNFnet(self.z0_hidden)
            odefunc = ODECNFfunc(
                diffeq=diffeq,
                divergence_fn="approximate",
                residual=False,
                rademacher=True,
            )
            cnf = CNF(
                odefunc=odefunc,
                T=1.0,
                train_T=False,
                regularization_fns=None,
                solver='dopri5',
                atol=1e-5, rtol=1e-5,
            )
            return cnf

        #CNF
        if self.CNF:
            print("------using CNF------")
            cnf_blocks = []
            for _ in range(1):
                cnf_blocks.append(construct_cnf())
            self.cnf_blocks = SequentialFlow(cnf_blocks).cuda()

        #output layer
        self.output = nn.Sequential(
                                    nn.Linear(self.output_in_dim, 128),
                                    nn.PReLU(),
                                    nn.Linear(128, 32),
                                    nn.PReLU(),
                                    nn.Linear(32, self.output_out_dim),
                                    nn.PReLU())


    def forward(self,s_inp,flow_x,flow_y,T,TP=None):        #s_inp [128, 168, 3]
        if self.flow_history:
            s_inp = torch.cat([s_inp,flow_x], dim =2)

        batch_size, hours_length, n_dims = s_inp.size()
        h3,all_h_hidden,z_hidden,all_z_hidden,kl_all,all_mean,all_std = run_rnn(self,s_inp,self.rnn_cell,self.z_cell,self.z_mean_ode,self.z_std_ode)
        rnn_input = torch.cat([all_h_hidden,all_z_hidden],dim=2)  #[128,178]

        #concatenate [Z,H]
        z4,h4 = self.rnn(rnn_input)
        gt = torch.cat([h4[0], h4[1]], dim=1)

        #extrapolation decoder
        zt  = self.decoder(z_hidden)            #[128, 168, 128][128, 128]

        #multi-feature fusion
        con = torch.cat([gt,zt],dim=1)

        #EFEN
        #using time information
        if self.use_time_info:
            week = self.embed_week(T[:, 0].long().view(-1, 1)).view(-1, 2)
            hour = self.embed_hour(T[:, 1].long().view(-1, 1)).view(-1, 3)
            weekend = torch.unsqueeze(T[:,2], 1)
            T_ode_in = self.fcn_1(torch.cat([week,hour,weekend], dim=1))
            T_out = self.T_ode(T_ode_in)
            con = torch.cat([con, T_out], dim=1)

        #using water flow information
        if self.use_ll_info:
            flow_ode_in = self.fcn_2(flow_y)
            flow_out = self.LL_ode_Y(flow_ode_in)
            con = torch.cat([con, flow_out], dim=1)

        #using temperature and electricity prices
        if self.temp_prices:
            TP_ode_in = self.fcn_3(TP)
            TP_ode = self.LL_ode_Y(TP_ode_in)
            con = torch.cat([con, TP_ode], dim=1)


        #reconstruct x_(t-1)
        rec_s_inp = self.reconstruct(z_hidden)

        #Output layer (MLP)
        pred = self.output(con)
        gaussian = Independent(Normal(loc=pred, scale=0.01), 1)
        return pred,gaussian,kl_all,rec_s_inp,#all_mean,all_std,z0,z_hidden,#,all_z_hidden


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

class SequentialFlow(nn.Module):
    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))
        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx