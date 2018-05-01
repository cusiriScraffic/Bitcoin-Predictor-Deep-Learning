from __future__ import unicode_literals, print_function, division
import pandas as pd
import datetime
from io import open
import glob
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from pprint import pprint
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#Sia Coin starts aug 31 2015
# Start Date Aug 07. 2015 because that is when ETH started getting data
numberOfOutChannels = 32
now = datetime.datetime.now()
year = str(now.year)
month = str(now.month)
day = str(now.day)
dateToday = year+month+day
cryptos =  ["bitcoin","ethereum","ripple","litecoin","stellar","monero",
"dash","nem","tether","verge","bytecoin-bcn","dogecoin"
,"bitshares","emercoin","maidsafecoin","nexus"] # Siacoin has a problem for some reason

lengthOfDataFrame = 968
trainingData = int(lengthOfDataFrame*0.80)
batchSize = 64

if len(month) == 1:
	month = "0"+month
if len(day) == 1:
	day = "0"+day

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv = nn.Conv2d(in_channels=6,out_channels=numberOfOutChannels,kernel_size=(1,2),stride=1,dilation=1).double()
		self.conv1 = nn.Conv2d(in_channels=numberOfOutChannels,out_channels=numberOfOutChannels,kernel_size=(1,2),stride=1,dilation=2).double()
		self.conv2 = nn.Conv2d(in_channels=numberOfOutChannels,out_channels=numberOfOutChannels,kernel_size=(1,2),stride=1,dilation=4).double()
		self.conv3 = nn.Conv2d(in_channels=numberOfOutChannels,out_channels=numberOfOutChannels,kernel_size=(1,2),stride=1,dilation=8).double()
		self.conv4 = nn.Conv2d(in_channels=numberOfOutChannels,out_channels=numberOfOutChannels,kernel_size=(1,2),stride=1,dilation=16).double()
		self.fc1 = nn.Linear(512,16).double()

	def forward(self,x):
		x = F.relu(self.conv(x))
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = x.squeeze(3)# On Third dim
		x = torch.cat(torch.unbind(x,2),1)# unbind dim 2 then cat together on dim 1
		x = self.fc1(x)
		x = nn.functional.tanh(x)
		return x

#Training network
cnn = Net()
criterion = nn.L1Loss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
matrix = np.zeros((6,len(cryptos),lengthOfDataFrame))# Create matrix of data
for i in range(len(cryptos)):
	url = "https://coinmarketcap.com/currencies/"+cryptos[i]+"/historical-data/?start=20150831&end="+dateToday
	df = pd.read_html(url, header=0)[0].drop(["Date"],axis=1).pct_change().drop([0])
	print(cryptos[i])
	open = list()
	close = list()
	high = list()
	low = list()
	mktCap = list()
	volume = list()
	for j in range(len(df)): # I drop the first line which is full of NAN
		open.append(df.iloc[j]["Open"])
		close.append(df.iloc[j]["Close"])
		high.append(df.iloc[j]["High"])
		low.append(df.iloc[j]["Low"])
		mktCap.append(df.iloc[j]["Market Cap"])
		volume.append(df.iloc[j]["Volume"])
	matrix[0,i,:] = open
	matrix[1,i,:] = close
	matrix[2,i,:] = high
	matrix[3,i,:] = low
	matrix[4,i,:] = mktCap
	matrix[5,i,:] = volume
for sample in range(10):
	input = np.zeros((batchSize,6,len(cryptos),32))
	target = np.zeros((batchSize,16))
	for batch in range(batchSize):
		start = np.random.randint(trainingData)
		singleInputBatching = matrix[:,:,start:start+32]
		input[batch,:,:,:] = singleInputBatching
		for i in range(len(cryptos)):
			target[batch,i] = matrix[1,i,start+32] #close,Which coin to predict,the next day
	input = torch.from_numpy(input).type(torch.DoubleTensor)
	cnn.zero_grad()
	# input = torch.unsqueeze(input,0)
	input = Variable(input) # Trying to remember the math but memory was being shared amoungst varible
	output = cnn(input)
	loss = criterion(output, Variable(torch.from_numpy(target).type(torch.DoubleTensor)))
	loss.backward(retain_graph=True) #Gradient
	optimizer.step() # Backprop		
	print("loss: "+str(loss.data[0]))#+"\t output: "+str(output.data[0,0])+"\tTarget: "+str(getPercentageChange(dictionaryOfCryptoObjects[crypto].close)[start+32]))
	print("-------------------------------")
print("\n\n\n")
print("-------------------------------")
print("Printing Predictions")
print("-------------------------------")
#Launching Predictions
batchSize = 1
cnn = Net()
for day in range(trainingData,lengthOfDataFrame-34):
	tmp = 0
	input = matrix[:,:,day:day+32]
	input = torch.from_numpy(input).type(torch.DoubleTensor)
	input = torch.unsqueeze(input,0)
	input = Variable(input) # Trying to remember the math but memory was being shared amoungst varibles
	output = cnn(input)
	target = np.zeros((1,16))
	for i in range(len(cryptos)):
			tmp = i
			target[0,i] = matrix[1,i,day+32] #close,Which coin to predict,the next day
	loss = criterion(output, Variable(torch.DoubleTensor(target)))
	print("loss: "+str(loss.data[0])+"\t output: "+str(output.data)+"\tTarget: "+str(target))	
	print("-------------------------------")
	print("Numpy",output.data[0].numpy().shape)
	print("target",target[0,:].shape)
	plt.title(cryptos[tmp])
	plt.plot(output.data[0].numpy(),target[0,:],'ro')
	plt.ylabel('Actual Close Percentage Change')
	plt.xlabel('Predicted Close Percentage Change')
plt.show()	

