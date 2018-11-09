#https://linlinzhao.github.io/tech/2017/10/21/Understanding-backward()-in-PyTorch.html

#First let’s recall the gradient computing under mathematical notions. For an independent 
#variable x (scalar or vector), the whatever operation on x is y=f(x). The gradient of 
#y w.r.t x_i's is ... 'the gradient of y'

#then plugging in a specific point of x = [X_1, X_2, ...] we'll get the gradient of y on 
#that point as a vector

#But pytorch's approach is confusing...
#1. mathematically we would say the gradient of function y wrt independent variables x
#but the gradients .grad is attached to the leaf variables x
#2. The parameter grad_variables of the function backward() is not straightforward
#3. What is retain_graph doing?

from __future__ import print_function
import torch
import torch.autograd
from torch.autograd import Variable
import numpy as np

'''
Define a scalar variable, set requires_grad to be true to add it to backward path 
for computing gradients

It is actually very simple to use backward()

first define the computation graph, then call backward()
'''

#x is a leaf created by user, thus grad_fn is None
x = Variable(torch.randn(1, 1), requires_grad=True)
print('x', x)

#define an operation on x
y = 2 * x
print('y', y)

#define one more operation to check the chain rule
z = y ** 3
print('z', z)

#The simple operations defined a forward path z=(2x)^3, z will be the final output Variable 
#we would like to compute gradient: dz=24x^2 dx, which will be passed to the parameter 
#Variables in backward() function.

#compute gradients, returns the gradient for the LEAVES
z.backward()

print('z gradient', z.grad) #None! as backward() only returns the gradient for leaves 
print('y gradient', y.grad) #None! as backward() only returns the gradient for leaves 
print('x gradient', x.grad) # note that x.grad is also a Variable

#x.grad should be interpreted as the gradient of z at x
#gradients for x are attached to x and not z, as it would be chaos if 
#x was a multi-dimensional vectorz`

#With flexibility in PyTorch’s core, it is easy to get the .grad for intermediate 
#Variables with help of register_hook function.

#How do we use grad_variables?
#Default value of grad_variables: torch.FloatTensor([1])
x = Variable(torch.randn(1, 1), requires_grad=True) #x is a leaf created by user, thus grad_fn is none
print('x', x)
y = 2 * x
z = y ** 3
z.backward(torch.FloatTensor([1]), retain_graph=True) #note that we are also retain_graph=True
print('Keeping the default value of grad_variables gives')
print('z gradient', z.grad)
print('y gradient', y.grad)
print('x gradient', x.grad)
#Change the value of grad_variables
x.grad.data.zero_()
z.backward(torch.FloatTensor([0.1]), retain_graph=True)
print('Modifying the default value of grad_variables to 0.1 gives')
print('z gradient', z.grad)
print('y gradient', y.grad)
print('x gradient', x.grad)

#Now lets set x to be a matrix!!!
x = Variable(torch.randn(2, 2), requires_grad=True) #x is a leaf created by user, thus grad_fn is none
print('x', x)
#define an operation on x
y = 2 * x
#define one more operation to check the chain rule
z = y ** 3
print('z shape:', z.size())
z.backward(torch.FloatTensor([1, 0]), retain_graph=True)
print('x gradient', x.grad)
x.grad.data.zero_() #the gradient for x will be accumulated, it needs to be cleared.
z.backward(torch.FloatTensor([0, 1]), retain_graph=True)
print('x gradient', x.grad)
x.grad.data.zero_()
z.backward(torch.FloatTensor([1, 1]), retain_graph=True)
print('x gradient', x.grad)
#We can see that  the gradients of z are computed w.r.t to each dimension of x, because 
#the operations are all element-wise. T.FloatTensor([1, 0]) will give the gradients for 
#first column of x.

#Then what if we render the output one-dimensional (scalar) while x is two-dimensional. 
#This is a real simplified scenario of neural networks.
x = Variable(torch.randn(2, 2), requires_grad=True) #x is a leaf created by user, thus grad_fn is none
print('x', x)
#define an operation on x
y = 2 * x
#print('y', y)
#define one more operation to check the chain rule
z = y ** 3
out = z.mean()
print('out', out)
out.backward(torch.FloatTensor([1]), retain_graph=True)
print('x gradient', x.grad)
x.grad.data.zero_()
out.backward(torch.FloatTensor([0.1]), retain_graph=True)
print('x gradient', x.grad)

#What is retain graph doing???
#When training a model, the graph will be re-generated for each iteration. 
#Therefore each iteration will consume the graph if the retain_graph is false, 
#in order to keep the graph, we need to set it be true.
x = Variable(T.randn(2, 2), requires_grad=True) #x is a leaf created by user, thus grad_fn is none
print('x', x)
#define an operation on x
y = 2 * x
#print('y', y)
#define one more operation to check the chain rule
z = y ** 3
out = z.mean()
print('out', out)
out.backward(T.FloatTensor([1]))  #without setting retain_graph to be true, this gives an error.
print('x gradient', x.grad)
x.grad.data.zero_()
out.backward(T.FloatTensor([0.1]))
print('x gradient', x.grad)

#Notes
#If you need to backward() twice on a graph or subgraph, you will need to set retain_graph 
#to be true, since the computation of graph will consume itself if it is false.
#Remember that gradient for Variable will be accumulated, zero it if do not need 
#accumulation.
