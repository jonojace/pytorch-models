#from https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html

#tensors are similar to ndarrays but can be used on GPU to accelerate computing
from __future__ import print_function
import torch

######################################################
# Tensors
######################################################

#construct uninitialised 5x3 matrix
x = torch.empty(5, 3)
# print(x) #encounter error when printing

#construct a randomly initialised matrix
x = torch.rand(5, 3)
print(x)

#construct matrix filled with zeros of datatype long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#construct a tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

#construct a 2D tensor directly from data
x = torch.tensor([[5.5, 3], [100, 101]])
print(x)

#create a tensor based on an existing tensor
#will reuse properties of the input tensor e.g. dtype
#unless new values are provided by the user
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)

#check the size (torch.Size is infact a tuple and will support all tuple operations)
print(x.size())

######################################################
# Operations
######################################################

#addition syntax 1
y = torch.rand(5, 3)
print(x + y)

#addition syntax 2
print(torch.add(x, y))

#addition syntax 3: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

#addition syntax 4: in-place 
#(NB any op that mutates a tensor in place is post fixed with an _)
#i.e. x.copy_(y) or x.t_()
y.add_(x)
print(y)

#You can use standard Numpy like indexing!
print(x)
print(x[:, 1]) #prints the second column

#Can resize or reshape tensor using torch.view
#Called view and not reshape as it doesn't modify the shape of the object
#Returns a different view on the object
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) #size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

#If tensor is one element, use .item() to get the value as a python num
x = torch.randn(1)
print(x)
print(x.item())

######################################################
# Numpy Bridge
######################################################

#Converting to and from torch tensors and numpy arrays is easy
#they share the same underlying memory locations so changing one 
#will change the other
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

#adding to the torch tensor also changes the numpy array
a.add_(1)
print(a)
print(b)

#Converting numpy array to torch tensor
#see how changing the np array changed the torch tensor automatically
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
#all the tensors on the CPU except a CharTensor support converting
#to numpy and back

######################################################
# CUDA tensors
######################################################

#Tensors can be moved onto any device using the .to method
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
else:
	print("Cuda not available")