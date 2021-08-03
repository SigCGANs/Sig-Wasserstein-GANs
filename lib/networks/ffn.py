from typing import Tuple
import torch.nn as nn
import torch



class FFN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int]):
        super().__init__()
        
        blocks = []
        input_dim_block = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(nn.Linear(input_dim_block, hidden_dim))
            blocks.append(nn.PReLU())
            input_dim_block = hidden_dim
        blocks.append(nn.Linear(input_dim_block, output_dim))
        self.network = nn.Sequential(*blocks)
        self.output_dim = output_dim

    def forward(self, *args):
        x = torch.cat(args, -1)
        out = self.network(x)
        return out



