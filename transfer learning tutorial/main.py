from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

'''
In practice, very few people train an entire Convolutional Network from 
scratch (with random initialization), because it is relatively rare to 
have a dataset of sufficient size. Instead, it is common to pretrain a 
ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million 
images with 1000 categories), and then use the ConvNet either as an 
initialization or a fixed feature extractor for the task of interest.

Finetuning the convnet: Instead of random initializaion, we initialize 
the network with a pretrained network, like the one that is trained on 
imagenet 1000 dataset. Rest of the training looks as usual.

ConvNet as fixed feature extractor: Here, we will freeze the weights for 
all of the network except that of the final fully connected layer. This 
last fully connected layer is replaced with a new one with random weights 
and only this layer is trained.
'''

plt.ion()   # interactive mode
