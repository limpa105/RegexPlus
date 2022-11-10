'''
Listener model, attempt 1: averaging stuff!

Input: a set of N examples
Algorithm:
 1. Encode each example with a (GRU) ENCODER
 2. Average the context vectors
 3. DECODER (GRU) has N ATTENTION HEADS, averaging the attention output vectors
'''

from typing import *
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_gru_cell import GRUCell

Floats = Ints = defaultdict(lambda: torch.Tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convention for dimensions:
# - B: Batch size
# - N: Number of examples
# - I: Input length (= max length of all the examples)
# - H: Hidden size

### Encoder!
class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.gru_cell = GRUCell(input_size, hidden_size)

    def forward(self, input: Ints['B×N×I'], input_lengths: Ints['B×N']) -> Tuple[Floats['B×N×I×H'], Floats['B×H']]:
        # type: (torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        '''
        Run the encoder on the WHOLE SEQUENCES
        Output:
         - the history of hiddens, B x N x I x H
         - the last hidden states, B x H
        '''
        B, N, I = input.size()
        H = self.hidden_size

        history = torch.empty(B,N,I,H, device=input.device)
        hidden = torch.zeros(B,N,H, device=input.device)

        for i in range(I):
            history[:,:,i] = hidden = self.gru_cell(input[:,:,i], hidden)

        hidden = history[torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0), input_lengths-1]  # BxNxH

        # Hidden state combination function: averaging
        return history, hidden.sum(dim=1) / N


### Attention weights!
class BilinearLocal(nn.Module):
    '''Local attention with predictive alignment, like from Luong et al'''
    def __init__(self, hidden_size: int, D=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.D = D
        self.sigma = D/2
        self.var_times_2 = 2 * self.sigma * self.sigma

        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        self.pos = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid())

    def forward(self,
            hidden: Floats['B×H'],
            encoder_hiddens: Floats['B×N×I×H'],
            input_lengths: Ints['B×N']
            ) -> Floats['B×N×I']:
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        B, N, I, H = encoder_hiddens.size()
        assert H == self.hidden_size
        p = self.pos(hidden) * input_lengths  # BxN
        idx = torch.arange(I, device=hidden.device)
        return self.bilinear(encoder_hiddens, hidden.unsqueeze(1).unsqueeze(1).expand(-1,N,I,-1)).squeeze(3) \
                - torch.square(idx-p.unsqueeze(2)) / self.var_times_2

class Bilinear(nn.Module):
    '''Plain bilinear attention'''
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

    def forward(self,
            hidden: Floats['B×H'],
            encoder_hiddens: Floats['B×N×I×H'],
            input_lengths: Ints['B×N']
            ) -> Floats['B×N×I']:
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        B, N, I, H = encoder_hiddens.size()
        assert H == self.hidden_size
        return self.bilinear(encoder_hiddens, hidden.unsqueeze(1).unsqueeze(1).expand(-1,N,I,-1)).squeeze(3)


### Decoder!
class Decoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, attn_weights):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn_weights = attn_weights

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,
            input: Ints['B'],
            hidden: Floats['B×H'],
            encoder_history: Floats['B×N×I×H'],
            input_lengths: Ints['B×N']
            ) -> Tuple[Floats['B × output size'], Floats['B×H'], Floats['B×N×I']]:
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        B, N, I, H = encoder_history.size()
        assert H == self.hidden_size

        attn_weights = self.attn_weights(hidden, encoder_history, input_lengths)  # BxNxI
        attn_weights.masked_fill_(torch.arange(I, device=input.device) >= input_lengths.unsqueeze(2), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=2)  # BxNxI 

        # attn_applied: BxH
        attn_applied = torch.matmul(
                attn_weights.unsqueeze(2),
                encoder_history,
                ).squeeze(2).sum(dim=1) / N

        # combined_input: BxH
        combined_input = F.relu(
                self.embedding(input) + self.attn_combine(attn_applied))
        hidden = self.gru(combined_input, hidden)  # BxH
        output = F.log_softmax(self.out(hidden), dim=1)
        return output, hidden, attn_weights



