#Neural networks can be constructed using the torch.nn package.

#Now that you had a glimpse of autograd, nn depends on autograd to define models and 
#differentiate them. An nn.Module contains layers, and a method forward(input)that 
#returns the output.

#digit recogniser network

#Training procedure:
#Define the neural network that has some learnable parameters (or weights)
#Iterate over a dataset of inputs
#Process input through the network
#Compute the loss (how far is the output from being correct)
#Propagate gradients back into the network’s parameters
#Update the weights of the network, typically using a simple update rule: 
#	weight = weight - learning_rate * gradient

#Define the network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		#Below we define the layers in the network without running them

		#1 input image channel, 6 output channels, 5x5 square convolution kernel
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)

		#an affine operation: y =  Wx + b, fully connected layers
		self.fc1 = nn.Linear(16 * 5 * 5, 120) #multiplied by 5 * 5 as we will flatten 
		#output of conv layer for tje fully connected layer 
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	#You just have to define the forward function, and the backward function 
	#(where gradients are computed) is automatically defined for you using autograd. 
	#You can use any of the Tensor operations in the forward function.
	def forward(self, x):
		#Below we pass input x through the layers in the order and manner we choose

		#Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

		#If the size is a square you can specify a single number to max_pool2d
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)

		#Flatten for fully connected layer
		x = x.view(-1, self.num_flat_features(x))

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x

	def num_flat_features(self, x):
		size = x.size()[1:] #all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

net = Net()
print(net)

#the learnable params of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

#lets try a random 32x32 input 
#note Expected input size to this net(LeNet) is 32x32. To use this net on MNIST dataset, 
#please resize the images from the dataset to 32x32.
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

#zero the gradient buffers of all params
net.zero_grad()
#backprop with random gradients
out.backward(torch.randn(1, 10))

'''
Note: Mini-batches

torch.nn only supports mini-batches. The entire torch.nn package only supports inputs 
that are a mini-batch of samples, and not a single sample.

For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.

If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
'''

'''
Recap

torch.Tensor - A multi-dimensional array with support for autograd operations like 
backward(). Also holds the gradient of output w.r.t. the tensor.

nn.Module - Neural network module. Convenient way of encapsulating parameters, with 
helpers for moving them to GPU, exporting, loading, etc.

nn.Parameter - A kind of Tensor, that is automatically registered as a parameter when 
assigned as an attribute to a Module.

autograd.Function - Implements forward and backward definitions of an autograd operation.
 Every Tensor operation, creates at least a single Function node, that connects to 
 functions that created a Tensor and encodes its history.
'''

'''
At this point, we covered:
	Defining a neural network
	Processing inputs and calling backward

Still Left:
	Computing the loss
	Updating the weights of the network
'''

'''
Loss function

A loss function takes the (output, target) pair of inputs, and computes a value that 
estimates how far away the output is from the target.

There are several different loss functions under the nn package . A simple loss is: 
nn.MSELoss which computes the mean-squared error between the input and the target.
'''
output = net(input)
target = torch.arange(1, 11) #dummy target for this example
print(target.size())
target = target.view(1, -1) #make it the same shape as output
print(target.size())

criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

'''
Now, if you follow loss in the backward direction, using its .grad_fn attribute, 
you will see a graph of computations that looks like this:

input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss

So, when we call loss.backward(), the whole graph is differentiated w.r.t. the loss, 
and all Tensors in the graph that has requires_grad=True will have their .grad Tensor 
accumulated with the gradient.
'''

#follow a few steps backward
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

'''
Backprop
To backpropagate the error all we have to do is to loss.backward(). You need to clear 
the existing gradients though, else gradients will be accumulated to existing gradients.
(as we called out.backward() beforehand)

Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and 
after the backward.
'''

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

'''
Update the weights

We use stochastic gradient descent: 
	weight = weight - learning_rate * gradient

Could implement using this simple code:
	learning_rate = 0.01
	for f in net.parameters():
	    f.data.sub_(f.grad.data * learning_rate)

But can also use torch.optim to make it simpler and also adds different update rules
'''

import torch.optim as optim 

#create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

#in your training loop
optimizer.zero_grad()   # zero the gradient buffers
output = net(input) #forward pass through network
loss = criterion(output, target) #calculate loss
loss.backward() #backprop loss through network
optimizer.step()    # Does the update of network weights using the gradients