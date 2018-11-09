'''
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#sphx-glr-intermediate-char-rnn-classification-tutorial-py
'''

'''
PREPARING THE DATA
'''

import glob
import os
import unicodedata #for converting from unicode to plain ASCII
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def findFiles(path): 
    '''wrapper to make glob name more intuitive'''
    return glob.glob(path)

#check if we can access our data
print(findFiles('data/names/*.txt'))

#create var holding valid characters and punctuation
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    '''convert unicode to ascii string and 
    only keep what we consider valid chars 
    as defined by all_letters'''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

#build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

#read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

#build list of categories (languages)
for filename in findFiles('data/names/*.txt'):
    #get the basename (filename + extension), but then leave out the extension
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines 
    
n_categories = len(all_categories)

'''
TURNING NAMES INTO TENSORS
'''

#letters are represented by one hot vectors
#dim: <1 x n_letters> 

#words are represented by multiple of these one hot vectors
#dim: <line_length x 1 x n_letters> 
#here the extra 1 dimension is the batch size, which pytorch requires
#we keep a batch size of 1 for now

def letterToIndex(letter):
    '''return the index of a letter in our vocab'''
    return all_letters.find(letter)

def letterToTensor(letter):
    '''just for demonstration, turn a letter into a 
    <1 x n_letters> tensor/vector'''
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

#turn a line/name into a <line_length x 1 x n_letters> tensor
#or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

'''
CREATING THE NETWORK
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''use params to initialise certain layers'''
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        #question: why do we have separate params for hidden and outputs?
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        #concat together the hidden activations from previous timestep 
        #and inputs from current timestep
        combined = torch.cat((inputs, hidden), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

#run a forward pass through the network
inputs = letterToTensor('A') #define input char at this timestep 
hidden = torch.zeros(1, n_hidden) #define the hidden state from previous time step
output, next_hidden = rnn(inputs, hidden) #perform forward pass

#we can use lineToTensor so we are not creating a new tensor for every step
#instead we just index the letter in the input that we want to use
inputs = lineToTensor('Albert')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(inputs[0], hidden)
print(output)

'''
PREPARING FOR TRAINING
'''

def categoryFromOutput(output):
    '''for interpretability: takes the conditional probability distribution
    output by our network, and returns the category index and name'''
    _, indices = output.topk(1) #top k returns top k elements
    category_i = indices[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))

def randomChoice(l):
    '''returns random element from a list with uniform prob'''
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    '''quick way to get a training example, random category and line'''
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

'''
TRAINING THE NETWORK 

Each loop of training will:

Create input and target tensors
Create a zeroed initial hidden state
Read each letter in and
Keep hidden state for next letter
Compare final output to target
Back-propagate
Return the output and loss
'''

#give it a bunch of examples, let it make guesses, tell it if its wrong/right

criterion = nn.NLLLoss()

learning_rate = 0.005 #if u set this too high, it might explode
#too low, it might not learn

def train(category_tensor, line_tensor):
    '''training loop
    category_tensor is true label for this example
    line_tensor is inputs for this example'''

    #init the 'dummy' hidden activations needed for the first timestep
    hidden = rnn.initHidden()

    #zero the gradients
    rnn.zero_grad()

    #perform forward prop through each timestep
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    #calc loss between output distribution and target labels
    loss = criterion(output, category_tensor)

    #backprop loss through network 
    loss.backward()

    #update weights
    #add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    
    return output, loss.item()

'''
RUN THE TRAINING LOOP WITH A BUNCH OF EXAMPLES
'''

#set training and training metrics params
n_iters = 100000
print_every = 5000
plot_every = 1000

#keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    '''return difference from now to some time in the past, 
    in minutes and seconds'''
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60 
    return '{}m {:4}s'.format(m, s)

#get time that we start training
start = time.time()

#perform training for n_iter iterations 
for iter in range(1, n_iters + 1):
    #get a training example
    category, line, category_tensor, line_tensor = randomTrainingExample()

    #pass example through network, and backprop its loss to update weights
    output, loss = train(category_tensor, line_tensor)

    #keep track of the loss
    current_loss += loss

    #print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ ({})'.format(category)

        #note we are printing a prediction and its label so we can 'sense' if our model is improving
        print('{} {}% ({}) {:.4} {} / {} {}'.format(iter, 
                                                    iter / n_iters * 100, 
                                                    timeSince(start), 
                                                    loss, 
                                                    line, 
                                                    guess, 
                                                    correct))

    #add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0                                                

plt.figure()
plt.plot(all_losses)
# plt.show()

'''
EVALUATING THE RESULTS
'''

#we will present a confusion matrix of 
#rows (target language labels)
#columns (guessed language)

#keep track of correct guesses in a confusion matrix 
confusion = torch.zeros(n_categories, n_categories)

#how many examples we want to make predictions for
n_confusion = 10000

#just return an output given a line (like train but without backprop of gradients)
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

#Go through a bunch of examples, and record which are correctly guessed
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

#normalise by dividing each row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum() #division will apply to each element in the row

#set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)
plt.ylabel('true language')
plt.xlabel('predicted language')

#set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
# plt.show()

'''
RUNNING ON USER INPUT
'''


def predict(input_line, n_predictions=5):
    '''use the trained model to predict most likely countries for a given input'''
    print('\n> {}'.format(input_line))
    with torch.no_grad(): #sets all requires_grad flags to False 
        #requires_grad flag allows one to exclude subgraphs from 
        #gradient computation to increase efficiency
        output = evaluate(lineToTensor(input_line))

        #get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')

#run loop that accepts user input and then returns the models predictions for it
user_input = ''
while user_input is not 'q':
    user_input = input('Please type in a name (type q to quit):')
    if user_input is not 'q':
        predict(user_input)