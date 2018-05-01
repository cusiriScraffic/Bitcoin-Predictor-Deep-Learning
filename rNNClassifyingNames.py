#Basically networks with loops in them allowing information to persist ie good for time series data because they are related to sequences and list
# One type of RNN is LSTM which stands for long short term memory which defeats the problem of using older information and not just recent
# Instead of LSTM having a single nn layer they have 4 interacting in a special way. There is a cell state that is kind of like a conveyor belt
#With two gates allowing informatino in or out but could also remain unchanged. They use a sigmoid layer with 0-1 value to control what goes through

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
from pprint import pprint
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import math
import time
 
def findFiles(path): return glob.glob(path)

path = "/Users/calvinusiri/Desktop/CS/machinelearning/pytorchScripts/data/data/names/*.txt"
print(findFiles(path))

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

#Turn a unicode string to plan ACSII
def  unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)
print(unicodeToAscii('Ślusàrski'))

#Build the category_lines dictionary, a list of names per language
category_lines = dict()
all_categories = list()

#Read files and split into lines
def readLines(filenames):
	lines = open(filenames,encoding='utf-8').read().strip().split('\n')
	return [unicodeToAscii(line) for line in lines]

for filenames in findFiles(path):
	category = filenames.split('/')[-1].split('.')[0]
	all_categories.append(category)
	lines = readLines(filenames)
	category_lines[category] = lines

n_categories = len(all_categories)

print(category_lines["Italian"][:5])

# Turning names into tensors to make use of them. To represent a single letter we use a hot one vector <1 * n letters>
#Find letter index from all_letters eg "a" = 0
def letterToIndex(letter):
	return all_letters.find(letter)

# turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
	tensor = torch.zeros(1, n_letters)
	tensor[0][letterToIndex(letter)] = 1
	return tensor 

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
	tensor = torch.zeros(len(line),1, n_letters)
	for li , letter in enumerate(line):
		tensor[li][0][letterToIndex(letter)] = 1
	return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())


# Creating the network
class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size
		# Hidden layers and output layers and hidden layer fed into next instance 
		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
		self.i2o = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
	def forward(self,input, hidden):
		combined = torch.cat((input, hidden),1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.softmax(output) # One way to calculate loss if you take diff from softmax and correct label
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1,self.hidden_size)) # Returns tensor

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = Variable(letterToTensor('A'))
#Why do we always need to zero it out each time
hidden = Variable(torch.zeros(1,n_hidden))

output, next_hidden = rnn(input, hidden)


input = Variable(lineToTensor('Albert'))
hidden = Variable(torch.zeros(1, n_hidden))
output, next_hidden = rnn.forward(input[0], hidden)

# As you can see the output is a <1 x n_categories> Tensor, where every item is the likelihood of that category (higher is more likely).
print(output)

#Training the model

def categoryFromOutput(output):
	top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
	category_i = top_i[0][0]
	return all_categories[category_i], category_i

print(categoryFromOutput(output))

def randomChoice(l):
	return l[random.randint(0, len(l) - 1)]
def randomTrainingExample():
	category = randomChoice(all_categories)
	line = randomChoice(category_lines[category])
	category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
	line_tensor = Variable(lineToTensor(line))
	return category, line, category_tensor, line_tensor

for i in range(10):
	category, line, category_tensor, line_tensor = randomTrainingExample()
	print('category =', category, '/ line =', line)

#Training the netork
# Gives the loss function, how wrong we are
criterion = nn.NLLLoss()

learning_rate = 0.005 # Needs to be just right, if its too high then it might explode if its too low then it might not learn.

def train(category_tensor, line_tensor):

	hidden = rnn.initHidden()
	#Zeros out the gradient for better reading
	rnn.zero_grad()
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i],hidden)
		loss = criterion(output, category_tensor)
		# backwards propagration
		loss.backward(retain_graph=True)
	# Add parameters' gradient to values, multiplied to learning rate, simulated annealing take big learning rate then reduce slowly
	for p in rnn.parameters():
		p.data.add_(-learning_rate, p.grad.data)

	return output , loss.data[0]

n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = int() # We want to use an int construct which is initiallzed at 0
all_losses = []
def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s/60)
	s -= m * 60
	return '%dm %sm'% (m, s) 

start = time.time()
for iter in range(1,n_iters + 1):
	category, line , category_tensor, line_tensor = randomTrainingExample()
	output, loss = train(category_tensor, line_tensor)
	current_loss += loss

	#Print iter number loss name and guess
	if iter % print_every == 0:
		guess, guess_i = categoryFromOutput(output)
		correct = '✓' if guess == category else '✗ (%s)' % category
		print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
	# Add current loss avg to list of losses
	if iter % plot_every == 0:
		all_losses.append(current_loss / plot_every)
		current_loss = 0

