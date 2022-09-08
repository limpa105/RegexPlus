from typing import *
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from hard_choice import HardChoice

MAX_LENGTH = 26

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Encoder!
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

##### Attention weights! (Pick one.)
class LocationBased(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size * 2, MAX_LENGTH)

    def forward(self, embedded_input, hidden, encoder_hiddens):
        # hidden: 1 × 1 × hidden_size
        # embedded_input: 1 × 1 × hidden_size
        # encoder_hiddens: length × hidden_size
        length = encoder_hiddens.size()[1]
        combined = torch.cat((embedded_input, hidden), dim=2)
        # combined: 1 × 1 × (2 * hidden_size)
        return F.softmax(self.linear(combined)[:,:,:length], dim=2)
        # output: 1 × 1 × length

class LocationBased2(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size * 2, MAX_LENGTH)

    def forward(self, embedded_input, hidden, encoder_hiddens):
        # hidden: 1 × 1 × hidden_size
        # embedded_input: 1 × 1 × hidden_size
        # encoder_hiddens: length × hidden_size
        length = encoder_hiddens.size()[1]
        combined = torch.cat((embedded_input, hidden), dim=2)
        # combined: 1 × 1 × (2 * hidden_size)
        return F.softmax(self.linear(combined), dim=2)[:,:,:length]
        # output: 1 × 1 × length

class DotProduct(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, embedded_input, hidden, encoder_hiddens):
        # hidden: 1 × 1 × hidden_size
        # embedded_input: 1 × 1 × hidden_size
        # encoder_hiddens: 1 × MAX_LENGTH × hidden_size
        return F.softmax(torch.inner(hidden.squeeze(0), encoder_hiddens), dim=2)
        # output: 1 × 1 × MAX_LENGTH

class Bilinear(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

    def forward(self, embedded_input, hidden, encoder_hiddens):
        # hidden: 1 × 1 × hidden_size
        # embedded_input: 1 × 1 × hidden_size
        # encoder_hiddens: 1 × MAX_LENGTH × hidden_size
        length = encoder_hiddens.size()[1]
        return F.softmax(
                self.bilinear(encoder_hiddens, hidden.expand(-1, length, -1)).view(1,1,-1), dim=2)
        # output: 1 × 1 × MAX_LENGTH



class Local(nn.Module):
    '''Local attention with predictive alignment, like from Luong et al'''
    def __init__(self, hidden_size: int, WeightsClass, D=3):
        super().__init__()
        self.D = D
        self.sigma = D/2
        self.hidden_size = hidden_size
        self.pos = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid())
        self.weights = WeightsClass(hidden_size)

    def forward(self, embedded_input, hidden, encoder_hiddens):
        length = encoder_hiddens.size()[1]
        p = self.pos(hidden).view(1,1,-1) * length
        idx = torch.arange(length, device=device)
        scaling_factor = torch.exp(- torch.square(idx-p) / (2*self.sigma*self.sigma))
        unnormalized_weights = scaling_factor * \
                self.weights(embedded_input, hidden, encoder_hiddens)
        return unnormalized_weights / torch.sum(unnormalized_weights)


##### Decoder! (Pick one.)
class SoftDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, AttnWeightsClass):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = AttnWeightsClass(hidden_size)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, encoder_hiddens):
        # input: idk
        # hidden: 1 × 1 × hidden_size
        # encoder_outputs: 1 × MAX_LENGTH × hidden_size
        # encoder_hiddens: 1 × MAX_LENGTH × hidden_size

        embedded_input = self.embedding(input).view(1, 1, -1)
        # embedded_input: 1 × 1 × hidden_size

        attn_weights = self.attn(embedded_input, hidden, encoder_hiddens)
        # attn_weights: 1 × 1 × MAX_LENGTH

        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        # attn_applied: 1 × 1 × hidden_size

        output = torch.cat((embedded_input, attn_applied), dim=2) # 1 × 1 × (2*hidden_size)
        output = self.attn_combine(output) # 1 × 1 × hidden_size
        output = F.relu(output)  # 1 × 1 × hidden_size
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden  # (1×output_size, 1×1×hidden_size)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class HardDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, AttnWeightsClass):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = AttnWeightsClass(hidden_size)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, encoder_hiddens):
        # input: idk
        # hidden: 1 × 1 × hidden_size
        # encoder_outputs: 1 × MAX_LENGTH × hidden_size
        # encoder_hiddens: 1 × MAX_LENGTH × hidden_size

        embedded_input = self.embedding(input).view(1, 1, -1)
        # embedded_input: 1 × 1 × hidden_size

        attn_weights = self.attn(embedded_input, hidden, encoder_hiddens)
        # attn_weights: 1 × 1 × MAX_LENGTH

        probs = attn_weights.squeeze(0).squeeze(0)
        attn_applied = HardChoice.apply(encoder_outputs.squeeze(0), probs)
        attn_applied = attn_applied.unsqueeze(0).unsqueeze(0)
        # attn_applied: 1 × 1 × hidden_size

        output = torch.cat((embedded_input, attn_applied), dim=2) # 1 × 1 × (2*hidden_size)
        output = self.attn_combine(output) # 1 × 1 × hidden_size
        output = F.relu(output)  # 1 × 1 × hidden_size
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden  # (1×output_size, 1×1×hidden_size)

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class NoAttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, AttnWeightsClass):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, encoder_hiddens):
        # input: idk
        # hidden: 1 × 1 × hidden_size
        # encoder_outputs: 1 × MAX_LENGTH × hidden_size
        # encoder_hiddens: 1 × MAX_LENGTH × hidden_size

        embedded_input = self.embedding(input).view(1, 1, -1)
        # embedded_input: 1 × 1 × hidden_size
        output, hidden = self.gru(embedded_input, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden  # (1×output_size, 1×1×hidden_size)

