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

def D3D(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]
    
    if model_depth == 121:
        model = DenseNet(
            TransitionBlock, # transition_block
            [6, 12, 24, 16], # layers
            get_inplanes(), # block_inplanes: [64, 128, 256, 512]
            **kwargs # others
        )
    elif model_depth == 169:
        model = DenseNet(
            TransitionBlock,
            [6, 12, 32, 32],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 201:
        model = DenseNet(
            TransitionBlock,
            [6, 12, 48, 32],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 264:
        model = DenseNet(
            TransitionBlock,
            [6, 12, 64, 48],
            get_inplanes(),
            **kwargs
        )
    
    return model

def T3D(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]
    
    if model_depth == 121:
        model = DenseNet(
            TemporalTransitionBlock, # transition_block
            [6, 12, 24, 16], # layers
            get_inplanes(), # block_inplanes: [64, 128, 256, 512]
            **kwargs # others
        )
    elif model_depth == 169:
        model = DenseNet(
            TemporalTransitionBlock,
            [6, 12, 32, 32],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 201:
        model = DenseNet(
            TemporalTransitionBlock,
            [6, 12, 48, 32],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 264:
        model = DenseNet(
            TemporalTransitionBlock,
            [6, 12, 64, 48],
            get_inplanes(),
            **kwargs
        )
    
    return model

def main():
    pass

if __name__ == '__main__':
    main()