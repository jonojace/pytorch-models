#https://stackoverflow.com/questions/43451125/pytorch-what-are-the-gradient-arguments

'''
For neural networks, we usually use loss to asses how well the network has learned to 
classify the input image(or other tasks). The  loss term is usually a scalar value. 
In order to update the parameters of the network, we need to calculate the gradient of 
loss w.r.t to the parameters, which is actually leaf node in the computation graph (by 
the way, these parameters are mostly the weight and bias of various layers such 
Convolution, Linear and so on).

According to chain rule, in order to calculate gradient of loss w.r.t to a leaf node, 
we can compute derivative of loss w.r.t some intermediate variable, and gradient of 
intermediate variable w.r.t to the leaf variable, do a dot product and sum all these up.

The gradient arguments of a Variable's backward() method is used to calculate a weighted 
sum of each element of a Variable w.r.t the leaf Variable. These weight is just the 
derivate of final loss w.r.t each element of the intermediate variable.
'''

from torch.autograd import Variable
import torch

x = Variable(torch.FloatTensor([[1, 2, 3, 4]]), requires_grad=True)
z = 2*x
loss = z.sum(dim=1)

# do backward for first element of z
z.backward(torch.FloatTensor([[1, 0, 0, 0]]))
print(x.grad.data) #tensor([[ 2.,  0.,  0.,  0.]]) --> dz_1/dx
x.grad.data.zero_() #remove gradient in x.grad, or it will be accumulated

# do backward for second element of z
z.backward(torch.FloatTensor([[0, 1, 0, 0]]))
print(x.grad.data) #tensor([[ 0.,  2.,  0.,  0.]]) --> dz_2/dx
x.grad.data.zero_()

# do backward for all elements of z, with weight equal to the derivative of
# loss w.r.t z_1, z_2, z_3 and z_4
z.backward(torch.FloatTensor([[1, 1, 1, 1]]))
print(x.grad.data) #tensor([[ 2.,  2.,  2.,  2.]]) --> 1*dz_1/dx + 1*dz_2/dx + 1*dz_3/dx + 1*dz_4/dx
x.grad.data.zero_()

#NB
#[1, 1, 1, 1] is exactly derivative of loss w.r.t to z_1, z_2, z_3 and z_4. 
#The derivative of loss w.r.t to x is calculated as:
#d(loss)/dx = d(loss)/dz_1 * dz_1/dx 
			#+ d(loss)/dz_2 * dz_2/dx 
			#+ d(loss)/dz_3 * dz_3/dx 
			#+ d(loss)/dz_4 * dz_4/dx

# or we can directly backprop using loss
loss.backward() # equivalent to loss.backward(torch.FloatTensor([1.0]))
print(x.grad.data) #tensor([[ 2.,  2.,  2.,  2.]])  
#So the output of 4th print is the same as the 3rd print