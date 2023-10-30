import numpy as np 

import torch
from torchvision import transforms

from data_loader import JesterV1

# Define a transformation without normalization
transform = transforms.Compose([transforms.ToTensor()])

# Load the dataset
data_dir = 'D:\Khanh\Others\Hand_Gesture\datasets\JESTER-V1'
dataset = JesterV1(
    data_dir = data_dir,
    num_frames = 30,
    transform = transform,
    mode = 'train',
    small = True
)

# Calculate mean and standard deviation
data = torch.stack([sample for sample, _ in dataset], dim = 0)
print(data.shape)
mean = data.mean(dim = (0, 2, 3))
std = data.std(dim = (0, 2, 3))
print('Mean:', mean)
print('Std:', std)