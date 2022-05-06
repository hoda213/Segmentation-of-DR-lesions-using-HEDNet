import sys
from torch.autograd import Variable
import os
import numpy as np
import random
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import config
from hednet import HNNNet
from utils import get_images
from dataset import FGADRDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset

# rotation_angle = config.ROTATION_ANGEL
rotation_angle = 0
image_size = config.IMAGE_SIZE
image_dir = config.IMAGE_DIR
lesion_name = config.LESION_NAME
batchsize = config.TRAIN_BATCH_SIZE

train_image_paths, train_mask_paths = get_images(image_dir, '7', 'train')
train_dataset = FGADRDataset(train_image_paths, train_mask_paths)
train_loader = DataLoader(train_dataset, batchsize, shuffle=True)
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    print (loader)
    for data , _ in loader:
        
        channels_sum += torch.mean(data, dim = [0,2,3])
        channels_squared_sum += torch.mean(data**2, dim = [0,2,3])
        num_batches +=1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std

mean , std = get_mean_std(train_loader)
print('mean:', mean)
print('std:', std)