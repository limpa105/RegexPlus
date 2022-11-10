'''
A GRU cell optimized for a very small input language

This is optimizable because you no longer need an embedding

Haven't actually verified that it works yet lol
'''

import torch
import torch.nn as nn

class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        assert input_size < hidden_size, "This is built for very small languages"

        self.r_emb = nn.Embedding(input_size, hidden_size)
        self.r_lin = nn.Linear(hidden_size, hidden_size, bias=False)
        self.z_emb = nn.Embedding(input_size, hidden_size)
        self.z_lin = nn.Linear(hidden_size, hidden_size, bias=False)
        self.n_emb = nn.Embedding(input_size, hidden_size)
        self.n_lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        '''
        Inputs:
          - input: A (...) tensor of integer indices in [0,input_size)
          - hidden: A (..., hidden_size) tensor of floats
        Output:
          - new_hidden: A (..., hidden_size) tensor of floats
        '''
        r = torch.sigmoid(self.r_emb(input) + self.r_lin(hidden))
        interp_factor = torch.sigmoid(self.z_emb(input) + self.z_lin(hidden))
        new_hidden = torch.tanh(self.n_emb(input) + r * self.n_lin(hidden))
        return (1-interp_factor) * new_hidden + interp_factor * hidden


