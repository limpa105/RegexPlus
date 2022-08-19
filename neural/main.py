import string
PAD_token = 0
SOS_token = 1
EOS_token = 2
MAX_LENGTH = 25
import torch
import random
from torch.autograd import Variable
from model import EncoderRNN, LuongAttnDecoderRNN
from masked_cross_entropy import *
import time
import math 
from torch import optim
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd 

#MAKING THE LANGUAGE TO USE
regex_only : list = ['[0-9]','[a-z]','[A-Z]','[a-zA-Z]', '[a-zA-Z0-9]', '[0-9]+','[a-z]+','[A-Z]+','[a-zA-Z]+', '[a-zA-Z0-9]+']
ascii_char : list = list(string.printable)[:95]
special_char = ['!','"','#','%','&',"'",',','-','.',':',';',
 '<','>','@','_','`',' ']
#regex_things = regex_things + [ i for i in ascii_char if i not in special_char]
regex_things = regex_only + ascii_char[:65] + special_char

#helper function
def split(example):
    return [char for char in example]

# Langauge class 
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS
      
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

#reading in the langauges 
def readLangs(lang1: list, lang2:list, reverse=False):
  input_lang = Lang('regex')
  output_lang = Lang('english')
  input_lang.addList(lang1)
  output_lang.addList(lang2)
  print(input_lang.name, input_lang.n_words)
  print(output_lang.name, output_lang.n_words)
  return input_lang, output_lang


def indexesFromList(lang, char_list):
    indexes = [lang.word2index[word] for word in char_list]
    indexes.append(EOS_token)
    return indexes


def tensorFromList(lang, char_list):
    indexes = indexesFromList(lang, char_list)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

"""
def pairFromPair(pair, input_lang):
    input_tensor = tensorFromList(input_lang, pair[0])
    #target_tensor = tensorFromList(output_lang, list(pair[1]))
    return (pair[0], input_tensor)

def tripleFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromList(input_lang, pair[0])
    target_tensor = tensorFromList(output_lang, pair[1])
    return (pair[0], input_tensor, target_tensor)
"""

def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) <= MAX_LENGTH and len(pair[1]) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs


def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

#reading in the data 
def read_in_data(test_data, train_regex, train_examples):
	test = pd.read_csv(test_data, sep='\n', header=None, names = ['regex'])
	train_single = pd.read_csv(train_regex, sep='\n', header=None, names = ['regex'])
	train_ten = train_single.loc[train_single.index.repeat(10)].reset_index(drop=True)
	with open(train_examples) as f:
		lines = f.readlines()
	examples = [split(line.strip()) for line in lines]
	train_ten["example"] = examples
	return (train_ten, test)

def random_batch(batch_size):
    # code inspired by: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        pair = random.choice(pairs)
        input_seqs.append(indexesFromList(input_lang, pair[0]))
        target_seqs.append(indexesFromList(output_lang, pair[1]))

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
        
    return input_var, input_lengths, target_var, target_lengths



def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, batch_size =10, max_length=MAX_LENGTH):
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    """
    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()
    """

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s // (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_epochs, batch_size = 10, print_every=1000, plot_every=100, learning_rate=0.01, optimizer='SGD', file=None):
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
    
    epoch = 0
    while(epoch< n_epochs):
        epoch+=1
        input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size)


        loss = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer)

    
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # Print out in some CSV format
            stats_test = accuracy_stats_for(encoder, decoder, dataset='testing')
            stats_train = accuracy_stats_for(encoder, decoder, dataset='training')
            output = f'{timeSince(start, epoch)},{epoch},{print_loss_avg},{stats_test},{stats_train}'
            print(output)
            if file: file.write(output + '\n')
            #print('%s (%d %d%%) %.4f' % (timeSince(start, epoch),
                                         #epoch, epoch/ n_epochs * 100, print_loss_avg))


def evaluate(encoder, decoder, input_seq, max_length=MAX_LENGTH):
    input_lengths = [len(input_seq)]
    input_seqs = [indexesFromList(input_lang, input_seq)]
    with torch.no_grad():
        input_batches = Variable(torch.LongTensor(input_seqs)).transpose(0, 1)
    
        
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    with torch.no_grad():
        decoder_input = Variable(torch.LongTensor([SOS_token])) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    
    # Store output words and attention states
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0].item()
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def accuracy_stats_for(encoder, decoder, dataset='testing', maxsize=1000):
    '''
    Returns "correct percent,reached end percent"
    '''
    if dataset == 'testing':
        iter = list(test_df.iterrows())[:maxsize]
    else:
        iter = list(train_df.iterrows())[:maxsize]
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

print("Starting data Set up")
train_df, test_df = read_in_data("data/test_data.txt", "data/train_data.txt", 'data/train_data_pairs.txt')
print("Starting to generate pairs")
pairs = [(eval(row["regex"]), row["example"]) for index,row in train_df.iterrows()]
pairs = filter_pairs(pairs)
print("Finished generate pairs")
input_lang, output_lang = readLangs(regex_things, ascii_char)


def train_neural_network(hidden_size=256, batch_size=10, dropout_p=0.1, optimizer='SGD', learning_rate =0.01):
    print(f'Training {hidden_size=} {dropout_p=} {optimizer=} {learning_rate=}...')
    file = open(f'Data for {hidden_size=} {dropout_p=} {optimizer=} {learning_rate=}.csv', 'w')
    file.write('time,iteration,loss,test accuracy,test end,train accuracy,train end\n')
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = LuongAttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, 1200000, batch_size, print_every=100, optimizer=optimizer, learning_rate=learning_rate, file=file)
    file.close()

#TODO: DEVELOP NEW TESTS WITH BATCHING 
setups = [
    { 'hidden_size': 256, 'dropout_p': 0.1, 'optimizer': 'SGD', 'learning_rate': 0.01 },
    { 'hidden_size': 256, 'dropout_p': 0.1, 'optimizer': 'SGD', 'learning_rate': 0.01 },
    { 'hidden_size': 256, 'dropout_p': 0.1, 'optimizer': 'SGD', 'learning_rate': 0.001 },
    { 'hidden_size': 128, 'dropout_p': 0.1, 'optimizer': 'SGD', 'learning_rate': 0.01 },
    { 'hidden_size': 512, 'dropout_p': 0.1, 'optimizer': 'SGD', 'learning_rate': 0.01 },
    { 'hidden_size': 256, 'dropout_p': 0.0, 'optimizer': 'SGD', 'learning_rate': 0.01 },
    { 'hidden_size': 256, 'dropout_p': 0.1, 'optimizer': 'Adam', 'learning_rate': 0.01 },
]

for setup in setups:
    train_neural_network(**setup)

