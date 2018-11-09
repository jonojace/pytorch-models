'''
autograd provides automatic diff for all ops on Tensors

it is a define-by-run framework

which means that your backprop is defined by how your code is run

that is. EVERY SINGLE ITERATION CAN BE DIFFERENT

Examples following...
'''

'''
Tensor

torch.Tensor is the central class of the package. If you set its attribute .requires_grad 
as True, it starts to track all operations on it. When you finish your computation you 
can call .backward() and have all the gradients computed automatically. The gradient 
for this tensor will be accumulated into .grad attribute.

To stop a tensor from tracking history, you can call .detach() to detach it from the 
computation history, and to prevent future computation from being tracked.

To prevent tracking history (and using memory), you can also wrap the code block in with 
torch.no_grad():. This can be particularly helpful when ***evaluating a model*** because the 
model may have trainable parameters with requires_grad=True, but for which we don’t need 
the gradients.
'''

'''
Function 

There’s one more class which is very important for autograd implementation - a Function.

Tensor and Function are interconnected and build up an acyclic graph, that encodes a 
complete history of computation. Each tensor has a .grad_fn attribute that references a 
Function that has created the Tensor (except for Tensors created by the user - their 
grad_fn is None).

If you want to compute the derivatives, you can call .backward() on a Tensor. If Tensor 
is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments 
to backward(), however if it has more elements, you need to specify a gradient argument 
that is a tensor of matching shape.
'''

import torch 

#create a tensor and set requires_grad=True to track computation with it
#NB we won't keep track of gradients on y or z
x = torch.ones(2, 2, requires_grad=True)
print(x)

#do an operation of tensor
y = x + 2
print(y)

#y was created as a result of an operation so it has a grad_fn
print(y.grad_fn)

#do more operations on y
z = y * y * 3
out = z.mean()
print(z, z.grad_fn)
print(out, out.grad_fn)


#.requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place. 
#The input flag defaults to True if not given.
#Q: why would we want to change the flag in-place???
a = torch.randn(2, 2) #create tensor
a = ((a * 3) / (a - 1)) #do operation
print(a)
print(a.requires_grad) #requires_grad is False, so its not tracking operations on it
a.requires_grad_(True)
print(a.requires_grad) #a now requires_grad
b = (a * a).sum() #b requires_grad
print(b.grad_fn, b.requires_grad) #a and b tracks the functions applied to it

'''
Gradients 

Lets backprop now

Because out contains a single scalar
out.backward() is equivalent to out.backward(torch.tensor(1))

backward() computes dout/dx for every param x that has requires_grad=True
they are accumulated onto x.grad for every parameter x

backward() accumulates gradients in the leaves (whatever vars have requires_grad=True)
you might need to zero them before calling it
'''
out.backward()

#print gradients
#d(out)/dx
print(x.grad)
#d(out)/dy
print(y.grad) #none as backward() only returns the gradient for leaves
#d(out)/dz
print(z.grad) #none as backward() only returns the gradient for leaves

#you can do many crazy things with autograd!
x = torch.rand(3, requires_grad=True)
y = x * 2 #perform 1 op
num_ops = 0
while y.data.norm() < 1000: #perform an undetermined number of ops
	y = y * 2
	num_ops += 1
print('num_ops', num_ops)
print(y)

#normally backward is called on a scalar loss func, so no need to pass gradients
#but since we call backward on y which is a vector of length 3 
#we have to supply it a vector of gradients of length 3
#these gradients are used to calculate a weighted sum of each element
#of a variable wrt the leaf variable
#for more info see understanding_backward.py
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)

#can stop autograd from tracking history on Tensors
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
	print((x ** 2).requires_grad)

