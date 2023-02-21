from typing import *
from datetime import datetime
import time, math, random, string, csv, functools, pickle, sys, os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Step 1. Instantiate the `Lang`uages!
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
    regex_only  = ['[0-9]','[a-z]','[A-Z]','[a-zA-Z]', '[a-zA-Z0-9]', '(+)', '(*)', 'OPT(', ')OPT'] #['[0-9]','[a-z]','[A-Z]','[a-zA-Z]', '[a-zA-Z0-9]', '[0-9]+','[a-z]+','[A-Z]+','[a-zA-Z]+', '[a-zA-Z0-9]+']
    ascii_char = list(string.printable)[:95]
    regex_things = regex_only + ascii_char

    regex = Lang('regex')
    text = Lang('text')
    regex.addList(regex_things)
    text.addList(ascii_char)
    print(regex.name, regex.n_words)
    print(text.name, text.n_words)
    return regex, text

regex_lang, text_lang = make_langs()

##### Step 2. Wrangling data
'''
pairs, test_regexes = pickle.load(open('../optionals/train-and-test-with-opts-v3.pickle', 'rb'))
_ , test_regexes1 = pickle.load(open('../optionals/train-and-test-with-opts.pickle', 'rb'))
_ , test_regexes2 = pickle.load(open('../optionals/train-and-test-with-opts-v2.pickle', 'rb'))
_ , test_regexes3 = pickle.load(open('../optionals/train-and-test-with-opts-v3.pickle', 'rb'))
train_pairs = pairs[:1000]
random.shuffle(pairs)
'''

def cat_and_pad(tensors):
    lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long, device=device)
    result = torch.zeros(len(tensors), max(lengths), dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        result[i,:lengths[i]] = tensors[i]
    return result, lengths

def pair_to_tensors(pair: Tuple[List[str], List[List[str]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def tensor_from_list(lang: Lang, char_list: List[str]) -> torch.Tensor:
        indexes = [lang.word2index[word] for word in char_list] + [EOS_token]
        return torch.tensor(indexes, dtype=torch.long, device=device)
    regex_tensor = tensor_from_list(regex_lang, pair[0])
    text_tensors = [tensor_from_list(text_lang, s) for s in pair[1]]
    examples_tensor, examples_lengths = cat_and_pad(text_tensors)
    return examples_tensor, examples_lengths, regex_tensor

def pairs_to_tensors(pairs):
    examples_tensors, examples_lengths, regex_tensors = zip(*(pair_to_tensors(p) for p in pairs))
    examples_tensor = torch.zeros(len(pairs), *max(t.size() for t in examples_tensors), dtype=torch.long, device=device)
    for i, t in enumerate(examples_tensors):
        examples_tensor[i,:,:t.size(1)] = t
    examples_lengths = torch.cat([l.unsqueeze(0) for l in examples_lengths], dim=0)
    regex_tensor, regex_lengths = cat_and_pad(regex_tensors)
    return examples_tensor, regex_tensor, examples_lengths, regex_lengths

### Step 3. Evaluation Metrics
"""
def sample(encoder, decoder, pair: List[str], strategy='sample'):
    '''Returns a regex for the input examples'''
    NUM_EXAMPLES = 1
    with torch.no_grad():
        B = 1  # Batch size
        example_tensor, regex_tensor, example_length, regex_lengths = pairs_to_tensors([pair])
        encoder_history, hidden = encoder(example_tensor, example_length)
        decoder_input = torch.tensor([SOS_token], device=device).expand(B)
        MAX_LEN = 100
        out = []
        weights = torch.zeros(example_length.size()[1], 100, example_length.max())
        index = 0
        while len(out) < MAX_LEN:
            decoder_output, hidden, attn_weights = decoder(
                    decoder_input,
                    hidden,
                    encoder_history,
                    example_length)
            if strategy == 'sample':
                word = np.random.choice(regex_lang.n_words, p = np.exp(decoder_output[0].detach().to("cpu").numpy()))
            elif strategy == 'top':
                word = decoder_output.data.topk(1)[1].item()
            else:
                raise Exception(f'Unknown strategy {strategy}')
            weights[:,index] = attn_weights.data[0]
            if word == EOS_token:
                return (out, weights)
            else:
                out.append(regex_lang.index2word[word])
            decoder_input = torch.tensor([word], device = device)
            index+=1
        return None 


def accuracy_metric(encoder, decoder, dataset = pairs, strategy = 'top'):
    '''Returns percent correct and percent ended'''
    count = 0
    ended = 0 
    syntax_bad = 0
    for pair in dataset:
        result = sample(encoder, decoder, pair, strategy=strategy)
        if result is not None:
            regex, _ = result
            ended+=1
            true_regex, examples = pair
            try:
                if all(next_chars.matches(regex, ex) for ex in examples):
                    count+=1   
            except:
                syntax_bad+=1
    return (count/len(dataset), ended/len(dataset),syntax_bad/len(dataset) )


### Step 4. Training

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

class Trainer:
    def __init__(self, encoder, decoder, identifier, file=None, col_file = None):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_opt = torch.optim.Adam(encoder.parameters(), lr=0.0001)
        self.decoder_opt = torch.optim.Adam(decoder.parameters(), lr=0.0001)
        self.loss_fn = nn.NLLLoss(reduction='none')
        self.file = file
        self.col_file = col_file
        self.identifier = identifier

    def train_batch(self,
            examples_tensor: Ints['B×N×I'],
            regex_tensor: Ints['B × target_length'],
            examples_lengths: Ints['B×N'],
            regex_lengths: Ints['B']):
        '''A single batch of training'''
        B, N, I = examples_tensor.size()
        B2, max_target_length = regex_tensor.size()
        assert B == B2

        self.encoder_opt.zero_grad()
        self.decoder_opt.zero_grad()

        # Encode the stuff!
        encoder_history, hidden = self.encoder(examples_tensor, examples_lengths)

        decoder_input = torch.tensor([SOS_token], device=device).expand(B)

        # Decode the stuff!
        loss = torch.zeros(B, device=device)
        for di in range(max_target_length):
            decoder_output, hidden, __ = self.decoder(
                    decoder_input,
                    hidden,
                    encoder_history,
                    examples_lengths)
            unmasked_loss = self.loss_fn(decoder_output, regex_tensor[:,di])
            loss += (di < regex_lengths) * unmasked_loss
            decoder_input = regex_tensor[:,di]

        total_loss = loss.sum()
        total_loss.backward()
        self.encoder_opt.step()
        self.decoder_opt.step()

        count = regex_lengths.sum()
        return total_loss.item() / count.item()
    
    def collection(self, encoder, decoder):
        v2 = accuracy_metric(encoder, decoder, test_regexes2, 'top')
        v1 = accuracy_metric(encoder, decoder, test_regexes1, 'top')
        v3 = accuracy_metric(encoder, decoder, test_regexes3, 'top')
        return (v1, v2, v3)

    def train_iters(self, B, n_iters, print_every=100, rand_num_ex=False, collect_data_every=500):
        start = time.time()
        loss_total = 0  # Reset every print_every

        for iter in range(1, n_iters + 1):
            batch_idx = ((iter-1)*B) % len(pairs)
            l = random.randint(1,5) if rand_num_ex and iter != 1 else 5
            batch = pairs_to_tensors([(r, e[:l]) for r, e in pairs[batch_idx:batch_idx + B]])

            loss_total += self.train_batch(*batch)

            if iter % print_every == 0:
                loss_avg = loss_total / print_every
                loss_total = 0
                output = f'{timeSince(start, iter/n_iters)},{iter},{loss_avg}, {accuracy_metric(self.encoder, self.decoder, train_pairs, "top")}'
                print(output)
                if self.file:
                    self.file.write(output + '\n')
                    self.file.flush()

            if iter % collect_data_every == 0:
                 data = f'{self.collection(self.encoder, self.decoder)}'
                 if self.col_file:
                    self.col_file.write(data + '\n')
                    self.col_file.flush()


def run(name, n_iters=25000, B=100):
    dt = datetime.now()

    identifier = f'{name} {dt}'
    print(identifier)
    log_file = open(f'logs/{identifier}.csv', 'w')
    col_file = open(f'collection/{identifier}.csv', 'w')

    hidden_size = 256
    encoder = torch.jit.script(Encoder(text_lang.n_words, hidden_size).to(device))
    decoder = torch.jit.script(Decoder(hidden_size, regex_lang.n_words, Bilinear(hidden_size)).to(device))

    trainer = Trainer(encoder, decoder, identifier, log_file, col_file)
    trainer.train_iters(B, n_iters)

    torch.save(encoder.state_dict(), f'encoders/{identifier}.pt')
    torch.save(decoder.state_dict(), f'decoders/{identifier}.pt')


if __name__ == '__main__':
    identifier = sys.argv[1]
    try:
        B = int(sys.argv[2])
    except:
        B = 75
        print(f'Using batch size {B}')
    run(identifier, B=B)
"""




