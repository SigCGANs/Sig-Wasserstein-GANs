import torch
from torch import nn
from typing import Tuple

from lib.networks.resfnn import ResFNN


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int):
        super(LSTMDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lstm(x)[0][:, -1:]
        x = self.linear(h)
        return x


class ResFNNDiscriminator(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: Tuple[int]):
        super(ResFNNDiscriminator, self).__init__()

        self.resfnn = ResFNN(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.resfnn(x.reshape(batch_size, -1))
