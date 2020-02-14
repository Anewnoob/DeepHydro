from utils.ODE import *
from utils.helper import init_network_weights,reparameterize,linspace_vector,split_last_dim,get_device
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence,Independent
from model.encoder_decoder import *

#This code is gatherd from  https://github.com/YuliaRubanova/latent_ode

class LatentODE(nn.Module):
    def __init__(self, input_dim, latent_dim, device=None   # input_dim = 1  latent_dim = 256
                 , n_gru_units=100,n_ode_gru_dims = 168):
        super(LatentODE, self).__init__()
        ode_rnn_encoder_dim = latent_dim
        self.ODESolver = ODESolver(ODERNNfunc(n_ode_gru_dims)).to(device)

        ode_blocks1 = []
        for _ in range(1):
            ode_blocks1.append(ODEBlock(ODEfunc(128),method='dopri5',rtol=1e-5,atol=1e-5))
        self.ode_blocks = nn.Sequential(*ode_blocks1)
        self.ode_gru = Encoder_z0_ODE_RNN(
            latent_dim=ode_rnn_encoder_dim,
            input_dim=input_dim,  #input and the mask
            z0_diffeq_solver=self.ODESolver,
            n_gru_units=n_gru_units,
            device=device).to(device)

        self.decoder = Decoder(6, 128)

        self.MLP = nn.Sequential(
            nn.Linear(latent_dim, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, input_dim)).to(device)

        init_network_weights(self.MLP)

    def forward(self, s_inp,flow_x, time_steps_to_predict=None,time_steps=None,
		mask = None, n_traj_samples = None, mode = None):
        s_inp = torch.cat([s_inp, flow_x], dim=2)
        #encoder
        time_steps = np.arange(0, 1.68, 0.01, dtype = float)
        time_steps = torch.tensor(time_steps).float()
        #print(data.shape)  torch.Size([128, 168, 1])
        mean_z, std_z, latent_ys= self.ode_gru.run_odernn(s_inp,time_steps,run_backwards=False)
        mean_z = mean_z.squeeze(0)
        std_z = std_z.squeeze(0)
        std_z = std_z.abs()
        z = reparameterize(mean_z,std_z)
        #kl loss
        z0_distr = Normal(mean_z, std_z)
        z0_prior = Normal(torch.Tensor([0.]).cuda(), torch.Tensor([1.]).cuda())
        kl = kl_divergence(z0_distr, z0_prior)

        s_res,hidden = self.decoder(s_inp,z)

        #prediction layer (ODE + MLP)
        output_in = self.ode_blocks(z)
        pred = self.output(output_in)

        gaussian = Independent(Normal(loc=pred, scale=0.01), 1)
        return pred,gaussian,kl,s_res



class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,
                 update_gate=None,
                 reset_gate=None,
                 new_state_net=None,
                 n_units=100):
        super(GRU_unit, self).__init__()

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

    def forward(self, y_mean, y_std, x, masked_update=False):
        y_concat = torch.cat([y_mean, y_std, x], -1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert (not torch.isnan(new_y).any())

        new_y_std = new_y_std.abs()
        return new_y, new_y_std

class Encoder_z0_ODE_RNN(nn.Module):
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver=None,
                 z0_dim=None, GRU_update=None,
                 n_gru_units=100,
                 device=torch.device("cpu")):

        super(Encoder_z0_ODE_RNN, self).__init__()
        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim,       #256,1
                                       n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

        self.transform_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, self.z0_dim * 2))
        init_network_weights(self.transform_z0)

    def forward(self, data, time_steps, run_backwards=True):

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if len(time_steps) == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
            xi = data[:, 0, :].unsqueeze(0)
            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
        else:
            last_yi, last_yi_std, _, = self.run_odernn(
                data, time_steps, run_backwards=run_backwards)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = split_last_dim(self.transform_z0(torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()

        return mean_z0, std_z0

    def run_odernn(self, data, time_steps,run_backwards=False):  #time_step = 1,2,3,4,...,N
        #print(data.shape)
        n_traj, n_tp, n_dims = data.size() #128,168,1
        extra_info = []
        device = get_device(data)
        prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
        prev_t, t_i = time_steps[0]-0.01 , time_steps[0]
        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50  #tensor(0.0334)

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []

        time_points_iter = range(0, len(time_steps))
        if run_backwards:
            time_points_iter = reversed(time_points_iter)
        for i in time_points_iter:
            if (prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i))
                inc = self.z0_diffeq_solver.odefunc(prev_t,prev_y) * (t_i - prev_t)  #d(ht)
                assert (not torch.isnan(inc).any())

                ode_sol = prev_y + inc   #new_h = h + d (ht)
                #print(prev_y.shape,inc.shape,ode_sol.shape)  #torch.Size([1, 128, 10]) torch.Size([1, 128, 10]) torch.Size([1, 128, 10])
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)


                assert (not torch.isnan(ode_sol).any())
            else:
                print("large than minimum step:",minimum_step)
                n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0, :] - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, :, 0, :] - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1, :]

            xi = data[:, i, :].unsqueeze(0)
            #print(yi_ode.shape,prev_std.shape,xi.shape) torch.Size([1, 128, 256]) torch.Size([1, 128, 256]) torch.Size([1, 128, 1])
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)
            prev_y, prev_std = yi, yi_std
            if i+1 < len(time_steps):
                prev_t, t_i = time_steps[i], time_steps[i+1]
            latent_ys.append(yi)

        latent_ys = torch.stack(latent_ys, 1)
        assert (not torch.isnan(yi).any())
        assert (not torch.isnan(yi_std).any())

        return yi, yi_std, latent_ys  # get z0