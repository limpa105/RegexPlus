from __future__ import unicode_literals, print_function, division

import string 
import random 
from numpy.random import choice
from scipy.stats import skewnorm
import pandas as pd

from io import open
import unicodedata
import string
import re
import random
import csv


import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import next_chars
USE_TRUE_PROB = False
USE_NFA = False
MAX_LENGTH = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split(word):
    return [char for char in word]

# reading in the test data 
test = pd.read_csv("data/test_data.txt", sep='\n', header=None, names = ['regex'])
train_single = pd.read_csv("data/train_data.txt", sep='\n', header=None, names = ['regex'])
train_ten = train_single.loc[train_single.index.repeat(10)].reset_index(drop=True)
with open('data/train_data_pairs.txt') as f:
    lines = f.readlines()
examples = [split(line.strip()) for line in lines]
train_ten["example"] = examples

# creating pairs 
if USE_TRUE_PROB:
  pairs = [(eval(row["regex"]),eval(row["regex"])) for i, row in train_ten.iterrows()]
else :
  print("Starting to generate pairs")
  pairs = [(eval(row["regex"]), row["example"]) for index,row in train_ten.iterrows()]
  print("Done generating pairs")

# make the language 
# first iteration = very restricted language no optionals no constants
regex_only : list = ['[0-9]','[a-z]','[A-Z]','[a-zA-Z]', '[a-zA-Z0-9]', '[0-9]+','[a-z]+','[A-Z]+','[a-zA-Z]+', '[a-zA-Z0-9]+']
ascii_char : list = list(string.printable)[:95]
special_char = ['!','"','#','%','&',"'",',','-','.',':',';',
 '<','>','@','_','`',' ']
#regex_things = regex_things + [ i for i in ascii_char if i not in special_char]
regex_things = regex_only + ascii_char[:65] + special_char

SOS_token = 0
EOS_token = 1

# constructing langiage using 
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
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
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

def indexesFromList(lang, char_list):
    return [lang.word2index[word] for word in char_list]


def tensorFromList(lang, char_list):
    indexes = indexesFromList(lang, char_list)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def pairFromPair(pair):
    input_tensor = tensorFromList(input_lang, pair[0])
    #target_tensor = tensorFromList(output_lang, list(pair[1]))
    return (pair[0], input_tensor)

def tripleFromPair(pair):
    input_tensor = tensorFromList(input_lang, pair[0])
    target_tensor = tensorFromList(output_lang, pair[1])
    return (pair[0], input_tensor, target_tensor)


def train(input_regex, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    #target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    #use_true_probabilities = False
    #if random.random() < teacher_forcing_ratio else False

    # # Teacher forcing: Feed the target as the next input
    # if USE_TRUE_PROB:
    #   if USE_NFA:
    #     state = next_chars.regex_to_nfa(input_regex)
    #     chosen_token = np.inf
    #     count = 0
    #     total = 0
    #     reached_end = False
    #     while chosen_token!=EOS_token and count < MAX_LENGTH:
    #     #print(count)
    #         count +=1
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #           decoder_input, decoder_hidden, encoder_outputs)
    #         poss_chars = next_chars.possible_next_chars(state)

    #         indexed_chars = indexesFromList(output_lang, poss_chars)
    #         true_tensor = torch.zeros(output_lang.n_words)
    #         true_tensor = true_tensor.to(device)
    #         if next_chars.end_token_is_allowed_here(state):
    #           indexed_chars.append(EOS_token)
    #         for i in indexed_chars:
    #           true_tensor[i] = 1/ len(indexed_chars) 
    #     # create a tensor that gives equal probabilities to next chars 
    #         loss += criterion(decoder_output, true_tensor.reshape(1, output_lang.n_words))
    #         chosen_token = random.choice(indexed_chars)
    #         next_tensor = torch.zeros(output_lang.n_words)
    #         next_tensor[chosen_token] = 1
    #         if (chosen_token == EOS_token):
    #             reached_end = True 
    #             break
    #         state = next_chars.consume_a_char(state, output_lang.index2word[chosen_token])
    #     #print(output_lang.index2word[chosen_token])
    #     #re.complie(#amount of regex we have consumed so far)
    #     # 0 or 1
    #     #chosen_tensor = torch.tensor(chosen_token)
    #         decoder_input = torch.tensor(chosen_token, device = device)  # Teacher forcing
    #   else:
    #     nfa = next_chars.regex_to_nfa(input_regex)
    #     dfa = next_chars.DFA(nfa)
    #     state = dfa.nodes[0]
    #     ch = np.inf
    #     count = 0
    #     reached_end = False
    #     while ch!='END' and count < MAX_LENGTH:
    #     #print(count)
    #         count +=1
    #         decoder_output, decoder_hidden, decoder_attention = decoder(
    #           decoder_input, decoder_hidden, encoder_outputs)
    #         poss_chars = list(state.transitions.keys())

    #         indexed_chars = indexesFromList(output_lang, poss_chars)
    #         true_tensor = torch.zeros(output_lang.n_words)
    #         true_tensor = true_tensor.to(device)
    #         #if next_chars.end_token_is_allowed_here(state):
    #           #indexed_chars.append(EOS_token)
    #         for index, i in enumerate(indexed_chars):
    #           true_tensor[i] = state.transitions[poss_chars[index]][1]
    #         true_tensor[EOS_token] = state.p_end
    #         loss += criterion(decoder_output, true_tensor.reshape(1, output_lang.n_words))
    #         ch = np.random.choice(np.array(list(state.transitions.keys()) + ['END']),
    #                 p=np.array(list(w for j, w in state.transitions.values()) +
    #                     [state.p_end]))
    #         if ch == 'END':
    #             reached_end = True 
    #             break
    #         # move the dfa
    #         state = dfa.nodes[state.transitions[ch][0]]
    #         # get index of chose character 
    #         chosen_index = output_lang.word2index[ch]
    #         next_tensor = torch.zeros(output_lang.n_words)
    #         next_tensor[chosen_index] = 1

    # else:
    reached_end = True 
    count =0
    target_length = target_tensor.size(0)
    #print(target_length)
    for di in range(target_length):
        count+=1
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        #print(decoder_output)
        #print(target_tensor[di])
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing
        if count >= MAX_LENGTH:
            reached_end = False 
        

    if reached_end:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / count

import time
import math


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


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01, optimizer='SGD', file=None):
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
    if USE_TRUE_PROB:
      training_tuple = [pairFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    else:
      training_tuple = [tripleFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        #print(training_tuple[iter - 1])
        ex_tuple = training_tuple[iter - 1]
        input_regex = ex_tuple[0]

        input_tensor = ex_tuple[1]
        if not USE_TRUE_PROB:
          output_tensor = ex_tuple[2]
        else: 
          output_tensor = None
        loss = train(input_regex, input_tensor, output_tensor,  encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
    
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # Print out in some CSV format
            stats_test = accuracy_stats_for(encoder, decoder, dataset='testing')
            stats_train = accuracy_stats_for(encoder, decoder, dataset='training')
            output = f'{timeSince(start, iter)},{iter},{print_loss_avg},{stats_test},{stats_train}'
            print(output)
            if file: file.write(output + '\n')
            #print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                             iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromList(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]



def accuracy_stats_for(encoder, decoder, dataset='testing', maxsize=1000):
    '''
    Returns "correct percent,reached end percent"
    '''
    if dataset == 'testing':
        iter = list(test.iterrows())[:maxsize]
    else:
        iter = list(train_single.iterrows())[:maxsize]
    size = len(iter)
    count = 0
    end = 0
    end_correct=0
    for index, row in iter:
        regex_list = eval(row["regex"])
        answer = evaluate(encoder, decoder, regex_list)
        if re.search('^' + ''.join(regex_list) + '$', ''.join(answer[0][:-1])):
            count+=1
            if answer[0][-1] == "<EOS>":
                end_correct+=1
            #print(f'CORRECT, regex: {"".join(regex_list)}, generated: {"".join(answer[0][:-1])}')
        else:
            pass
            #print(f'WRONG, regex: {"".join(regex_list)}, generated: {"".join(answer[0][:-1])}')
        if answer[0][-1] == "<EOS>":
            end+=1
    return f'{count/size * 100},{end/size * 100}'

def train_neural_network(hidden_size=256, optimizer='SGD', learning_rate=0.01, iters=100000):
    print(f'Training {hidden_size=} {optimizer=} {learning_rate=}...')
    file = open(f'Data for {hidden_size=} {optimizer=} {learning_rate=}.csv', 'w')
    file.write('time,iteration,loss,test accuracy,test end,train accuracy,train end\n')
    file.flush()
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)
    trainIters(encoder1, attn_decoder1, iters, print_every=1000, optimizer=optimizer, learning_rate=learning_rate, file=file)
    file.close()

setups = [
    { 'hidden_size': 256, 'optimizer': 'Adam', 'learning_rate': 0.0003 },
    # { 'hidden_size': 256, 'optimizer': 'SGD', 'learning_rate': 0.01 },
]

for setup in setups:
    train_neural_network(**setup)

