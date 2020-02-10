# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
    print("use odeint_adjoint method")
else:
    from torchdiffeq import odeint
    print("use odeint method")

#######################################CNF############################################
class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODECNFnet(nn.Module):

    def __init__(self, inplanes,units = 128):
        super(ODECNFnet, self).__init__()
        #self.softplus1 = nn.Softplus()
        #self.softplus2 = nn.Softplus()
        self.elu1 = nn.ELU(inplace=True)
        self.elu2 = nn.ELU(inplace=True)
        self.fc1 = ConcatLinear(inplanes, units)
        self.fc2 = ConcatLinear(units, units)
        self.fc3 = ConcatLinear(units, inplanes)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(t,x)
        out = self.elu1(out)
        out = self.fc2(t,out)
        out = self.elu2(out)
        out = self.fc3(t,out)
        return out


#######################################ODE-NORMAL############################################
class ODEfunc(nn.Module):

    def __init__(self, inplanes,fn = nn.ELU(inplace=True),units = 128):
        super(ODEfunc, self).__init__()
        self.fn = fn
        self.fc1 = nn.Linear(inplanes, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, inplanes)
        self.nfe = 0


    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.fn(out)
        out = self.fc2(out)
        out = self.fn(out)
        out = self.fc3(out)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc,integration_time = torch.tensor([0, 1]).float(),method = 'fixed_adams',rtol=1e-5,atol=1e-5):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = integration_time
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,method=self.method)#fixed_adams
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value



#######################################ODE-RNN############################################
class ODERNNfunc(nn.Module):

    def __init__(self, inplanes,hidden = 100):
        super(ODERNNfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(inplanes, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, inplanes)
        self.nfe = 0


    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

class ODESolver(nn.Module):

    def __init__(self, odefunc):
        super(ODESolver, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self,x,time_steps_to_predict):
        #self.integration_time = time_steps_to_predict.type_as(x)
        out = odeint(self.odefunc, x, time_steps_to_predict, rtol=1e-3, atol=1e-4,method="fixed_adams")#fixed_adams
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

