from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

'''
https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html#sphx-glr-intermediate-char-rnn-generation-tutorial-py
'''

'''
Data preprocessing
'''

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))

'''
Define the network 

Compared to the last tutorial,
we add extra linear layers and a dropout layer
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        #define the layers that will be used in this network
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size +
                             hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size +
                             hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, inputs, hidden):
        #perform forward pass through network
        #note that category is an input this time
        #as we condition the output of names with a language
        input_combined = torch.cat((category, inputs, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

'''
Training helper functions
'''

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
#i.e. Kasparov -> Kasparov
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
#add a EOS marker onto end of this name/line
#i.e. Kasparov -> asparovEOS
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

'''Training the network
In contrast to classification, where only the last output is used, we are making a prediction at every step, so we are calculating loss at every step.
The magic of autograd allows you to simply sum these losses at each step and call backward at the end.
'''

criterion = nn.NLLLoss()

learning_rate = 0.0005 

def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0 

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

#initialise the network
rnn = RNN(n_letters, 128, n_letters)

#set training + metrics params
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0  # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    _, loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' %
              (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


plt.figure()
plt.plot(all_losses)
plt.show()

'''
Sampling the network

Give the network a letter, ask what the next one is, feed that in as 
the next letter and repeat until the EOS token
'''

max_length = 20

#sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad(): #no need to track history in sampling
        category_tensor = categoryTensor(category)
        inputs = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, inputs[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1: #EOS token
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            inputs = inputTensor(letter)
    
    return output_name

#Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    print('\nGenerating names for category {}'.format(category))
    for start_letter in start_letters:
        print(sample(category, start_letter))


samples('Russian')

samples('German')

samples('Spanish')

samples('Chinese')
