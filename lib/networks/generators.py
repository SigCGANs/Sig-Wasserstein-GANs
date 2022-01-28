from abc import abstractmethod

import signatory
import torch
import torch.nn as nn
import numpy as np
from lib.augmentations import apply_augmentations, get_number_of_channels_after_augmentations, parse_augmentations
from lib.augmentations import AddTime
from lib.utils import init_weights
from lib.networks.resfnn import ResFNN as ResFNN
from lib.networks.ffn import FFN


class GeneratorBase(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GeneratorBase, self).__init__()
        """ Generator base class. All generators should be children of this class. """
        self.input_dim = input_dim
        self.output_dim = output_dim

    #@abstractmethod
    def forward_(self, batch_size: int, n_lags: int, device: str):
        """ Implement here generation scheme. """
        #...
        pass

    def forward(self, batch_size: int, n_lags: int, device: str):
        x = self.forward_(batch_size, n_lags, device)
        x = self.pipeline.inverse_transform(x)
        return x


class LSTMGenerator(GeneratorBase):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, n_layers: int, init_fixed: bool = True):
        super(LSTMGenerator, self).__init__(input_dim, output_dim)
        # LSTM
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first = True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
        self.linear.apply(init_weights)
        # neural network to initialise h0 from the LSTM
        self.initial_nn = nn.Sequential(ResFNN(input_dim, hidden_dim*n_layers, [hidden_dim, hidden_dim]), nn.Tanh()) # we put a tanh at the end because we are initialising h0 from the LSTM, that needs to take values between [-1,1]
        self.initial_nn.apply(init_weights)
        self.init_fixed = init_fixed
        self.initial_nn.apply(init_weights)

    def forward(self, batch_size: int, n_lags: int, device: str) -> torch.Tensor:
        z = (0.1 * torch.randn(batch_size, n_lags, self.input_dim)).to(device)#cumsum(1)
        z[:,0,:] *= 0 # first point is fixed
        z = z.cumsum(1)
        
        if self.init_fixed:
            h0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
        else:
            z0 = torch.randn(batch_size, self.input_dim, device=device)
            h0 = self.initial_nn(z0).view(batch_size, self.rnn.num_layers, self.rnn.hidden_size).permute(1,0,2).contiguous()
        c0 = torch.zeros_like(h0)
        h1, _ = self.rnn(z, (h0, c0))
        x = self.linear(h1)
        
        assert x.shape[1] == n_lags
        return x



def compute_multilevel_logsignature(brownian_path: torch.Tensor, time_brownian: torch.Tensor, time_u: torch.Tensor, time_t: torch.Tensor, depth: int):
    """

    Parameters
    ----------
    brownian_path: torch.Tensor
        Tensor of shape [batch_size, L, dim] where L is big enough so that we consider this 
    time_brownian: torch.Tensor
        Time evaluations of brownian_path
    time_u: torch.Tensor
        Time discretisation used to calculate logsignatures
    time_t: torch.Tensor
        Time discretisation of generated path
    depth: int
        depth of logsignature

    Returns
    -------
    multi_level_signature: torch.Tensor

    ind_u: List
        List of indices time_u used in the logsigrnn
    """
    logsig_channels = signatory.logsignature_channels(in_channels = brownian_path.shape[-1], depth=depth)

    multi_level_log_sig = [] #torch.zeros(brownian_path.shape[0], len(time_t), logsig_channels)

    u_logsigrnn = []
    last_u = -1
    for ind_t, t in enumerate(time_t[1:]):
        u = time_u[time_u < t].max()
        ind_low = torch.nonzero((time_brownian<=u).float(), as_tuple=False).max() 
        if u != last_u:
            u_logsigrnn.append(u)
            last_u = u

        ind_max = torch.nonzero((time_brownian<=t).float(), as_tuple=False).max() 
        interval = brownian_path[:, ind_low:ind_max+1, :]
        multi_level_log_sig.append(signatory.logsignature(interval, depth=depth, basepoint=True))
    multi_level_log_sig = [torch.zeros_like(multi_level_log_sig[0])] + multi_level_log_sig

    #logsig_channels = signatory.logsignature_channels(in_channels = brownian_path.shape[-1], depth=depth)

    #multi_level_log_sig = [] #torch.zeros(brownian_path.shape[0], len(time_t), logsig_channels)

    #u_logsigrnn = []
    #last_u = -1
    #for ind_t, t in enumerate(time_t):
    #    u = time_u[time_u <= t].max()
    #    ind_low = torch.nonzero((time_brownian<=u).float(), as_tuple=False).max() 
    #    if u != last_u:
    #        u_logsigrnn.append(u)
    #        last_u = u

    #    ind_max = torch.nonzero((time_brownian<=t).float(), as_tuple=False).max() 
    #    interval = brownian_path[:, ind_low:ind_max+1, :]
    #    #if t == 0:
    #    multi_level_log_sig.append(signatory.logsignature(interval, depth=depth, basepoint=True))
    #    #else:
    #    #    multi_level_log_sig[:,ind_t] = signatory.logsignature(interval, depth=depth)

    return multi_level_log_sig, u_logsigrnn





class LogSigRNNGenerator(GeneratorBase):
    def __init__(self, input_dim, output_dim, n_lags, augmentations, depth, hidden_dim, n_layers, len_noise=1000, len_interval_u=50, init_fixed: bool = True):
        
        
        super(LogSigRNNGenerator, self).__init__(input_dim, output_dim)
        input_dim_rnn = get_number_of_channels_after_augmentations(input_dim, augmentations)

        logsig_channels = signatory.logsignature_channels(in_channels=input_dim_rnn, depth=depth)

        self.depth = depth
        self.augmentations = augmentations
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.len_noise = len_noise
        self.time_brownian = torch.linspace(0,1,self.len_noise)  # len_noise is high enough so that we can consider this as a continuous brownian motion
        self.time_u = self.time_brownian[::len_interval_u]
        # self.time_t = torch.linspace(0,1,n_lags)
        
        # definition of LSTM + linear at the end
        self.rnn = nn.Sequential(
                      FFN(input_dim = hidden_dim + logsig_channels, 
                          output_dim = hidden_dim,
                          hidden_dims = [hidden_dim, hidden_dim]),
                      nn.Tanh()
                    )
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

        self.rnn.apply(init_weights)
        self.linear.apply(init_weights)
        
        # neural network to initialise h0 from the LSTM
        self.initial_nn = nn.Sequential(ResFNN(input_dim, hidden_dim, [hidden_dim, hidden_dim]), nn.Tanh()) 
        self.initial_nn.apply(init_weights)
        self.init_fixed = init_fixed

    def forward(self, batch_size: int, n_lags: int, device: str,):
        time_t = torch.linspace(0,1,n_lags).to(device)
        z = torch.randn(batch_size, self.len_noise, self.input_dim, device=device)
        h = (self.time_brownian[1:] - self.time_brownian[:-1]).reshape(1,-1,1).repeat(1,1,self.input_dim)
        h = h.to(device)
        z[:,1:,:] *= torch.sqrt(h) 
        z[:,0,:] *= 0 # first point is fixed
        brownian_path = z.cumsum(1)
        y = apply_augmentations(brownian_path, self.augmentations)
        y_logsig, u_logsigrnn = compute_multilevel_logsignature(brownian_path=y, time_brownian=self.time_brownian.to(device), time_u=self.time_u.to(device), time_t = time_t.to(device), depth=self.depth)
        u_logsigrnn.append(time_t[-1])

        if self.init_fixed:
            h0 = torch.zeros(batch_size, self.hidden_dim).to(device)
        else:
            z0 = torch.randn(batch_size, self.input_dim, device=device)
            h0 = self.initial_nn(z0)

        last_h = h0
        x = torch.zeros(batch_size, n_lags, self.output_dim, device=device)
        for idx, (t, y_logsig_) in enumerate(zip(time_t, y_logsig)):
            h = self.rnn(torch.cat([last_h, y_logsig_],-1))
            if t >= u_logsigrnn[0]:
                del u_logsigrnn[0]
                last_h = h
            x[:,idx,:] = self.linear(h)

        assert x.shape[1] == n_lags
        return x

