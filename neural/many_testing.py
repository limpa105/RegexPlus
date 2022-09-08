from typing import *
from datetime import datetime
import time, math, random, string, csv, functools, pickle
import sys, os
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import regex_probs
from next_chars import error_location

from many_models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### Step 1. Instantiate the `Lang`uages!
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


##### Step 2. Wrangling data
NUM_EXAMPLES = 1
pairs, test_regexes = pickle.load(open('hmmmmm.pickle', 'rb'))
pairs = [(r,e[:NUM_EXAMPLES]) for r,e in pairs]

def tensorFromList(lang, char_list):
    indexes = [lang.word2index[word] for word in char_list]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tripleFromPair(pair):
    input_tensor = tensorFromList(input_lang, pair[0])
    target_tensor = functools.reduce(
            lambda a, b: torch.cat((a,b), dim=0), (tensorFromList(output_lang, s) for s in pair[1]))
    return (pair[0], input_tensor, target_tensor)


##### Step 3. Data collection

constants_dataset = pickle.load(open('constants.pickle', 'rb'))
short_dataset, long_dataset = pickle.load(open('short_long.pickle', 'rb'))

def edit_distance(thing_one, thing_two):
    if len(thing_one) == 0: return len(thing_two)
    if len(thing_two) == 0: return len(thing_one)
    mat = np.zeros((len(thing_one), len(thing_two)))
    for i in range(len(thing_one)):
        mat[i, 0] = i
    for j in range(len(thing_two)):
        mat[0, j] = j
    for i in range(1, len(thing_one)):
        for j in range(1, len(thing_two)):
            if thing_one[i] == thing_two[j]:
                mat[i, j] = mat[i-1, j-1]
            else:
                mat[i, j] = 1 + min(mat[i-1, j], mat[i, j-1], mat[i-1, j-1])
    return mat[-1,-1]


def avg_edit_distance(encoder, decoder, dataset, num_examples): 
    distances = 0
    count = 0
    for i in range(num_examples):
        answer1 = sample(encoder, decoder, dataset[i])
        answer2 = sample(encoder, decoder, dataset[i])
        if answer1 is None or answer2 is None: continue
        distances += edit_distance(answer1, answer2)
        count += 1
    return distances/count


def collection(encoder, decoder):
    test_stats = accuracy_metric(encoder, decoder, test_regexes)
    constant_stats = accuracy_metric(encoder, decoder, constants_dataset)
    short_stats = accuracy_metric(encoder, decoder, short_dataset)
    long_stats = accuracy_metric( encoder, decoder, long_dataset)
    ed = avg_edit_distance(encoder, decoder, test_regexes, 100)
    funky_g = funky_graph_stats(encoder, decoder, test_regexes)
    return (test_stats, constant_stats, short_stats, long_stats, ed, funky_g) 

def funky_graph_stats(encoder, decoder, dataset, iters=10):
    yes = np.zeros(MAX_LENGTH)
    nah = np.zeros(MAX_LENGTH)
    for regex in dataset:
        for i in range(iters):
            ex = sample(encoder, decoder, regex)
            if ex is None or len(ex) > MAX_LENGTH: continue
            i = error_location(regex, ex)
            if i == -1:
                yes[:len(ex)+1]+=1
            else:
                yes[:i] +=1
                nah[i] +=1
    distribution = (nah+0.5)/(yes+nah+1)
    return (np.mean(distribution) , np.var(distribution))

def accuracy_metric(encoder, decoder, dataset):
    '''Returns percent correct and percent ended'''
    count = 0
    ended = 0 
    for regex in dataset:
        answer = sample(encoder, decoder, regex)
        if answer is not None:
            ended+=1
            if regex_probs.Regex(regex).matches(answer):
                count+=1
    return (count/len(dataset), ended/len(dataset))
                


def sample(encoder, decoder, regex: List[str], max_length=MAX_LENGTH):
    '''Returns an example, for the input regex'''
    NUM_EXAMPLES = 1
    with torch.no_grad():
        input_tensor = tensorFromList(input_lang, regex)
        input_length = input_tensor.size()[0]
        hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(1, input_length, encoder.hidden_size, device=device)
        encoder_hiddens = torch.zeros(1, input_length, encoder.hidden_size, device=device)



        for ei in range(input_length):
            encoder_output, hidden = encoder(input_tensor[ei], hidden)
            encoder_outputs[:,ei] = encoder_output[:, 0]
            encoder_hiddens[:,ei] = hidden[:, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        out = ''
        
        MAX_LEN = 100
        while len(out) < MAX_LEN:
            decoder_output, hidden = decoder(decoder_input, hidden, encoder_outputs, encoder_hiddens)
            word = np.random.choice(output_lang.n_words, p = np.exp(decoder_output[0].detach().to("cpu").numpy()))
            if word == EOS_token:
                return out 
            else:
                out += output_lang.index2word[word]
            decoder_input = torch.tensor([[word]], device = device)
        return None 




##### Step 4. Training a model!!!

LOSS_FN = nn.CrossEntropyLoss()

def train(input_regex, input_tensor, target_tensor, encoder, decoder, encoder_opt, decoder_opt):
    '''A single iteration of training.'''
    hidden = encoder.initHidden()

    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # FIXME: should it be input_length or MAX_LENGTH ?
    encoder_outputs = torch.zeros(1, input_length, encoder.hidden_size, device=device)
    encoder_hiddens = torch.zeros(1, input_length, encoder.hidden_size, device=device)

    loss = 0

    # Encode the stuff!!
    for ei in range(input_length):
        encoder_output, hidden = encoder(input_tensor[ei], hidden)
        encoder_outputs[:,ei] = encoder_output[:,0]
        encoder_hiddens[:,ei] = hidden[:,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Decode the stuff!!
    count = target_length = target_tensor.size(0)
    for di in range(target_length):
        decoder_output, hidden = decoder(
                decoder_input, hidden, encoder_outputs, encoder_hiddens)
        loss += LOSS_FN(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()

    return loss.item() / count


def timeSince(since, percent):
    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, collect_data_every=10000, learning_rate=0.0003, optimizer=optim.Adam, file=None):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optimizer(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optimizer(decoder.parameters(), lr=learning_rate)
    training_tuple = [tripleFromPair(random.choice(pairs))
                      for i in range(n_iters)]

    fancy_data = []

    for iter in range(1, n_iters + 1):
        input_regex, input_tensor, output_tensor = training_tuple[iter - 1]
        loss = train(input_regex, input_tensor, output_tensor,  encoder,
                     decoder, encoder_optimizer, decoder_optimizer)

        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # Print out in some CSV format
            accuracy = accuracy_metric(encoder, decoder, test_regexes)
            output = f'{timeSince(start, iter/n_iters)},{iter},{print_loss_avg},{accuracy}'
            print(output)
            if file: file.write(output + '\n')

        if iter % collect_data_every == 0:
            # collecting data .....
            data = collection(encoder, decoder) 
            fancy_data.append(data)
            print(f'Collected fancy data! {data}')

    return fancy_data

def run(AttnWeightsClass, DecoderClass, name=None, n_iters=100000):
    dt = datetime.now()
    if name is None:
        name = f'{AttnWeightsClass.__name__} {DecoderClass.__name__}'

    dirname = f'Results {name} {dt}'
    os.mkdir(dirname)
    log_file = open(dirname + '/log.csv', 'w')

    hidden_size = 256
    optimizer = optim.Adam
    learning_rate = 0.0003
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderClass(hidden_size, output_lang.n_words, AttnWeightsClass).to(device)

    fancy_data = trainIters(encoder, decoder, n_iters, learning_rate=learning_rate, optimizer=optimizer, file=log_file)

    torch.save(encoder.state_dict(), dirname + '/encoder.pt')
    torch.save(decoder.state_dict(), dirname + '/decoder.pt')
    pickle.dump(fancy_data, open(dirname + '/fancy_data.pickle', 'wb'))
    

# Options:
# - Bilinear, DotProduct, LocationBased, LocationBased2
# - SoftDecoder, HardDecoder
# TODO: implement monotonic attention

#weights = eval(sys.argv[1])
#decoder = eval(sys.argv[2])

#run(weights, decoder, name=sys.argv[3])


