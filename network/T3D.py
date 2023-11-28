import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from functools import partial

def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride = 1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        bias = False
    )
    
def conv1x1x1(in_planes, out_planes, stride = 1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size = 1,
        stride = stride,
        bias = False
    )
    
class DenseBlock(nn.Module):
    pass 

class TransitionBlock(nn.Module):
    pass

class TemporalTransitionBlock(nn.Module):
    pass

class DenseNet(nn.Module):
    pass

class Temporal3DConvs(nn.Module):
    pass

def DenseNet(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]