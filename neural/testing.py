from typing import *
from datetime import datetime
import time, math, random, string, csv, functools, pickle
import sys

import regex_probs

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 26

NUM_EXAMPLES = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# reading in the data
pairs, test_regexes = pickle.load(open('hmmmmm.pickle', 'rb'))
pairs = [(r,e[:NUM_EXAMPLES]) for r,e in pairs]

# make the language
# first iteration = very restricted language no optionals
regex_only : list = ['[0-9]','[a-z]','[A-Z]','[a-zA-Z]', '[a-zA-Z0-9]', '[0-9]+','[a-z]+','[A-Z]+','[a-zA-Z]+', '[a-zA-Z0-9]+']
ascii_char : list = list(string.printable)[:95]
regex_things = regex_only + ascii_char

SOS_token = 0
EOS_token = 1

# constructing language using
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

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

def readLangs(lang1: list, lang2:list, reverse=False):
  input_lang = Lang('regex')
  output_lang = Lang('english')
  input_lang.addList(lang1)
  output_lang.addList(lang2)
  print(input_lang.name, input_lang.n_words)
  print(output_lang.name, output_lang.n_words)
  return input_lang, output_lang

input_lang, output_lang = readLangs(regex_things, ascii_char)

# model code also taken from
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
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
        self.attn = nn.Linear(self.hidden_size * 2, max_length)

        # Bad attn (did not work):
        # self.attn = nn.Bilinear(self.hidden_size, self.hidden_size, 1)

        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, encoder_hiddens):
        embedded = self.embedding(input).view(1, 1, -1)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), dim=2)), dim=2)

        # Bilinear attention (did not work):
        # length = encoder_outputs.size()[1]
        # attn_weights = F.softmax(
        #     self.attn(encoder_hiddens, hidden.expand(-1, length, -1)), dim=2)
        # attn_weights = attn_weights.squeeze(2).unsqueeze(1)

        # Dot product attention!
        # length = encoder_outputs.size()[1]
        # attn_weights = F.softmax(encoder_hiddens.squeeze(0).matmul(hidden.squeeze(0).squeeze(0))).unsqueeze(0).unsqueeze(0)
        #print(attn_weights.size())

        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        output = torch.cat((embedded, attn_applied), 2)
        output = self.attn_combine(output)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def tensorFromList(lang, char_list):
    indexes = [lang.word2index[word] for word in char_list]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tripleFromPair(pair):
    input_tensor = tensorFromList(input_lang, pair[0])
    target_tensor = functools.reduce(
            lambda a, b: torch.cat((a,b), dim=0), (tensorFromList(output_lang, s) for s in pair[1]))
    return (pair[0], input_tensor, target_tensor)


def train(input_regex, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(1, MAX_LENGTH, encoder.hidden_size, device=device)
    encoder_hiddens = torch.zeros(1, MAX_LENGTH, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[:,ei] = encoder_output[:,0]
        encoder_hiddens[:,ei] = encoder_hidden[:,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    reached_end = True
    count = 0
    target_length = target_tensor.size(0)
    #print(target_length)
    for di in range(target_length):
        count+=1
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, encoder_hiddens)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing
        if count >= MAX_LENGTH:
            reached_end = False


    if reached_end:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.0003, optimizer='Adam', file=None):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    if optimizer == 'SGD':
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    else:
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_tuple = [tripleFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        input_regex, input_tensor, output_tensor = training_tuple[iter - 1]
        loss = train(input_regex, input_tensor, output_tensor,  encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # Print out in some CSV format
            accuracy = accuracy_stats_for(encoder, decoder)
            output = f'{timeSince(start, iter/n_iters)},{iter},{print_loss_avg},{accuracy}'
            print(output)
            if file: file.write(output + '\n')

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    return plot_losses

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromList(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(1, MAX_LENGTH, encoder.hidden_size, device=device)
        encoder_hiddens = torch.zeros(1, MAX_LENGTH, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[:,ei] = encoder_output[:, 0]
            encoder_hiddens[:,ei] = encoder_hidden[:, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_examples = []
        decoded_word = ''

        for di in range(max_length * NUM_EXAMPLES):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, encoder_hiddens)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_examples.append(decoded_word)
                decoded_word = ''
                if len(decoded_examples) == NUM_EXAMPLES: break
            else:
                decoded_word += output_lang.index2word[topi.item()]

            decoder_input = topi.squeeze().detach()

        return decoded_examples


def accuracy_stats_for(encoder, decoder):
    '''
    Returns percent correct
    '''
    count = 0
    total = 0
    for regex in test_regexes:
        answer = evaluate(encoder, decoder, regex)
        for ex in answer:
            total += 1
            if regex_probs.Regex(regex).matches(ex):
                count += 1
    return f'{count/total * 100},{total/len(test_regexes)}'

def train_neural_network(hidden_size=256, optimizer='Adam', learning_rate=0.0003, iters=70000):
    dt = datetime.now()
    print(f'Training {hidden_size=} {optimizer=} {learning_rate=} at {dt}...')
    file = open(f'log-{optimizer}-lr={learning_rate}-{dt}.csv', 'w')
    file.write('time,iteration,loss,test accuracy,avg num examples\n')
    file.flush()
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    trainIters(encoder1, attn_decoder1, iters, print_every=1000, optimizer=optimizer, learning_rate=learning_rate, file=file)
    file.close()
    torch.save(encoder1.state_dict(), f'encoder {dt}.pt')
    torch.save(attn_decoder1.state_dict(), f'decoder {dt}.pt')

def funky_training(iters=50000):
    hidden_size = 256
    optimizer = 'Adam'
    learning_rate = 0.0003
    dt = datetime.now()
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    print(f'Training {hidden_size=} {optimizer=} {learning_rate=} at {dt}...')
    file = open(f'log-{optimizer}-lr={learning_rate}-{dt}.csv', 'w')
    file.write('time,iteration,loss,test accuracy\n')
    file.flush()
    global NUM_EXAMPLES
    global pairs
    saved_pairs = pairs
    for i in range(1,6):
        iters = [0, 50000, 30000, 20000, 20000, 10000][i]
        NUM_EXAMPLES = i
        pairs = [(r,e[:NUM_EXAMPLES]) for r,e in saved_pairs]
        trainIters(encoder, decoder, iters, print_every=1000, optimizer=optimizer, learning_rate=learning_rate, file=file)
        torch.save(encoder.state_dict(), f'encoder {dt} part {NUM_EXAMPLES}.pt')
        torch.save(decoder.state_dict(), f'decoder {dt} part {NUM_EXAMPLES}.pt')
    file.close()

setups = [
    { 'hidden_size': 256, 'optimizer': 'Adam', 'learning_rate': 0.003 },
    # { 'hidden_size': 256, 'optimizer': 'SGD', 'learning_rate': 0.01 },
]

for setup in setups:
    train_neural_network(**setup)

# funky_training()
# retrain_from_file('encoder 2022-08-30 23:19:29.074725.pt', 'decoder 2022-08-30 23:19:29.074725.pt')

