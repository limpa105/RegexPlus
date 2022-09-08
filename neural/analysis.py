
from typing import *
from datetime import datetime
import time, math, random, string, csv, functools, pickle, sys
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import regex_probs

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


MAX_LENGTH = 26 # should not be changed since it's a model parameter and we're loading the model from files

def LENGTH_DISTRIB(k): # TODO: make this be the right thing
    '''Return an (unnormalized) probability for the string to have length k'''
    # Poisson with mean 4
    return 4**k / np.math.factorial(k)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
      
    def addList(self, list_words):
        for word in list_words:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def make_langs():
    regex_only = ['[0-9]','[a-z]','[A-Z]','[a-zA-Z]', '[a-zA-Z0-9]', '[0-9]+','[a-z]+','[A-Z]+','[a-zA-Z]+', '[a-zA-Z0-9]+']
    ascii_char = list(string.printable)[:95]
    regex_things = regex_only + ascii_char

    input_lang = Lang('regex')
    output_lang = Lang('text')
    input_lang.addList(regex_things)
    output_lang.addList(ascii_char)
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang

# regex, text
input_lang, output_lang = make_langs()

def indexesFromList(lang, char_list):
    return [lang.word2index[word] for word in char_list]

def tensorFromList(lang, char_list):
    indexes = indexesFromList(lang, char_list)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


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


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # attn_weights = F.softmax(torch.zeros(1, self.max_length, device=device), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



### OK the actual stuff now

class Translator:
    def prob_of(self, regex: List[str], s: str) -> float:
        raise Exception('abstract method')
    def sample_from(self, regex: List[str]) -> str:
        raise Exception('abstract method')

class GroundTruth(Translator):
    def matches(self, regex: List[str], s: str) -> bool:
        return regex_probs.Regex(regex, 2*MAX_LENGTH, LENGTH_DISTRIB).matches(s)
    def prob_of(self, regex: List[str], s: str) -> float:
        return regex_probs.Regex(regex, 2*MAX_LENGTH, LENGTH_DISTRIB).prob_of(s)
    def sample(self, regex: List[str]) -> str:
        return regex_probs.Regex(regex, 2*MAX_LENGTH, LENGTH_DISTRIB).sample()

class Network:
    def __init__(self):
        # dec, enc = 'decoder 2022-08-31 20:31:25.242188 part 5.pt', 'encoder 2022-08-31 20:31:25.242188 part 5.pt'
        self.encoder = EncoderRNN(input_lang.n_words, 256)
        self.decoder = AttnDecoderRNN(256, output_lang.n_words)
        # self.encoder.load_state_dict(torch.load('saved-models/long-funky/encoder 2022-08-31 14:17:18.340388 part 5.pt'))
        self.encoder.load_state_dict(torch.load('encoder 2022-08-31 20:31:25.242188 part 1.pt'))
        self.encoder = self.encoder.to(device=device)
        # self.decoder.load_state_dict(torch.load('saved-models/long-funky/decoder 2022-08-31 14:17:18.340388 part 5.pt'))
        self.decoder.load_state_dict(torch.load('decoder 2022-08-31 20:31:25.242188 part 1.pt'))
        self.decoder = self.decoder.to(device=device)
        self.encoder.train(False)
        self.decoder.train(False)

    def prob_of(self, regex: List[str], s: str) -> float:
        input_tensor = tensorFromList(input_lang, regex)
        input_length = input_tensor.size()[0]

        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=device)

        hidden = self.encoder.initHidden()
        for i in range(input_length):
            encoder_out, hidden = self.encoder(input_tensor[i], hidden)
            encoder_outputs[i] += encoder_out[0,0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)

        tokens = tensorFromList(output_lang, list(s))
        log_prob = 0.0
        for i in range(tokens.size()[0]):
            log_prob += decoder_output[0][tokens[i][0]]
            decoder_output, hidden, _ = self.decoder(tokens[i], hidden,
                    encoder_outputs)
        return math.exp(log_prob)

    def sample_lots(self, regex: List[str]) -> Sequence[str]:
        input_tensor = tensorFromList(input_lang, regex)
        input_length = input_tensor.size()[0]

        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=device)

        hidden = self.encoder.initHidden()
        for i in range(input_length):
            encoder_out, hidden = self.encoder(input_tensor[i], hidden)
            encoder_outputs[i] += encoder_out[0,0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        out = ''
        while True:
            decoder_output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            word = np.random.choice(output_lang.n_words, p=np.exp(decoder_output[0].detach().to("cpu").numpy()))
            if word == EOS_token:
                yield out
                out = ''
            else:
                out += output_lang.index2word[word]
            decoder_input = torch.tensor([[word]], device=device)

        return None # Did not end
        # return out[:-1]

    def sample(self, regex: List[str]) -> str:
        for r in self.sample_lots(regex):
            return r


    def sample_lots_with_attention(self, regex: List[str]) -> Sequence[str]:
        input_tensor = tensorFromList(input_lang, regex)
        input_length = input_tensor.size()[0]

        encoder_outputs = torch.zeros(MAX_LENGTH, self.encoder.hidden_size, device=device)

        hidden = self.encoder.initHidden()
        for i in range(input_length):
            encoder_out, hidden, = self.encoder(input_tensor[i], hidden)
            encoder_outputs[i] += encoder_out[0,0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        regex_len = len(regex)+2
        out = ''
        weights = torch.zeros(100, regex_len+10)
        index = 0
        while True:
            decoder_output, hidden, attn_weights = self.decoder(decoder_input, hidden, encoder_outputs)
            word = np.random.choice(output_lang.n_words, p=np.exp(decoder_output[0].detach().to("cpu").numpy()))
            if word == EOS_token:
                weights[index] = attn_weights.data[0][:regex_len+10]
                return (out, weights[:index+1])
                out = ''

            else:
                out += output_lang.index2word[word]
                weights[index] = attn_weights.data[0][:regex_len+10]
                index+=1
            decoder_input = torch.tensor([[word]], device=device)

        return None # Did not end

gt = GroundTruth()
nn = Network()
print(gt.prob_of(['Q','[a-z]+'], 'Qat'))
print(nn.prob_of(['Q','[a-z]+'], 'Qat'))
print(gt.sample(['Q','[a-z]+']))
print(nn.sample(['Q','[a-z]+']))

train_data, regexes = pickle.load(open('hmmmmm.pickle', 'rb'))

def KL(p: Translator, q: Translator, samples_per_regex=1) -> float:
    count = 0
    acc = 0.
    for regex in regexes:
        for i in range(samples_per_regex):
            sample = p.sample(regex)
            if sample is None: continue
            count += 1
            acc += np.log2(p.prob_of(regex, sample) / q.prob_of(regex, sample))
    return acc / count


def diversity(regex):
    NUM_SAMPLES = 5
    samples = []
    while len(samples) < NUM_SAMPLES:
        sample = nn.sample(regex)
        if sample is not None and sample != '': samples.append(sample)
    common_chars = reduce(lambda x, y: x & y, (set(s) for s in samples))
    return sum(len(common_chars) / len(set(s)) for s in samples) / NUM_SAMPLES


def take(n, it):
    l = []
    for i in it:
        l.append(i)
        if len(l) >= n:
            return l

def by_length(data=long_regexes):
    counts = np.zeros(MAX_LENGTH)
    yeah = np.zeros(MAX_LENGTH)
    for r in data:
        counts[len(r)] += 1
        sample = nn.sample(r)
        if sample is not None and gt.matches(r, sample):
            yeah[len(r)] += 1
    return yeah, counts

def funky_stats(iters=10, data=long_regexes):
    from next_chars import error_location
    from matplotlib import pyplot as plt
    import scipy.stats

    # Gather data
    yes = np.zeros(MAX_LENGTH)
    nah = np.zeros(MAX_LENGTH)
    for regex in data:
        for i in range(iters):
            ex = nn.sample(regex)
            if len(ex) > MAX_LENGTH: continue
            i = error_location(regex, ex)
            if i == -1:
                yes[:len(ex)+1] += 1
            else:
                yes[:i] += 1
                nah[i] += 1

    # Do stats: find the expected probability and the 90% credible region
    lo = np.zeros(MAX_LENGTH)
    mid = np.zeros(MAX_LENGTH)
    hi = np.zeros(MAX_LENGTH)
    for i in range(MAX_LENGTH):
        dist = scipy.stats.beta(0.5 + nah[i], 0.5 + yes[i])
        lo[i], hi[i] = dist.interval(0.90)
        mid[i] = dist.mean()

    # Make a plot
    plt.clf()
    plt.plot(lo)
    plt.plot(mid)
    plt.plot(hi)
    plt.xlabel('index into example')
    plt.ylabel('P(error at that index)')
    print('now you should run plt.savefig("your file name.png")')
    return yes, nah

#yes, nah = funky_stats(iters=1, data=[r for r, __ in train_data[:100000]])
#plt.savefig('one-error-rate.png')



def see_attention(input_seq):
     output, attention_weights = nn.sample_lots_with_attention(input_seq)
     fig = plt.figure()
     ax = fig.add_subplot()
     print(len(input_seq))
     print(attention_weights.shape)
     print(len(output))
     cax = ax.matshow(attention_weights.numpy(),cmap ='bone')
     ax.set_xticklabels(['s'] + input_seq + ['<EOS>'])
     ax.set_yticklabels(['s'] + list(output) + ['<EOS>'])
     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
     
     #plt.show(block=True)

print(nn.sample([*'mark is cool']))
see_attention(['[a-z]+'])
#see_attention(['[a-z]+', 'K', '[0-9]+'])
plt.savefig('see_attention.png')
plt.show(block=True)

