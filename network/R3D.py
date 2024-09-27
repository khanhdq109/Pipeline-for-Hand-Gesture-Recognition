import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torchsummary import summary

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
    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride = 1, downsample = None):
        super().__init__()
        
        self.conv1 = conv3x3x3(in_planes, planes, stride) 
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace = True)
        
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.stride = stride
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out
    
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride = 1, downsample = None):
        super().__init__()
        
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        
        self.stride = stride
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out
    
class NLBlock(nn.Module):
    
    def __init__(self, in_planes, subsample = False):
        super().__init__()
        
        self.conv_theta = conv1x1x1(in_planes, in_planes // 2)
        self.conv_phi = conv1x1x1(in_planes, in_planes // 2)
        self.conv_g = conv1x1x1(in_planes, in_planes // 2)

        self.conv_z = conv1x1x1(in_planes // 2, in_planes)
        self.bn = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        
        self.subsample = subsample
        if self.subsample:
            self.pool = nn.MaxPool3d(kernel_size = 2, stride = 2, padding = 1)
        
    def forward(self, x):
        B, C, T, H, W = x.size()
        residual = x
        
        theta = self.conv_theta(x).view(B, C // 2, -1) # (B, C // 2, THW)
        phi = self.conv_phi(x).view(B, C // 2, -1) # (B, C // 2, THW)
        g = self.conv_g(x).view(B, C // 2, -1) # (B, C // 2, THW)
        
        if self.subsample:
            phi = self.pool(phi)
            g = self.pool(g)
        
        theta = theta.permute(0, 2, 1) # (B, THW, C // 2)
        attention = torch.matmul(theta, phi) # (B, THW, THW)
        attention = torch.softmax(attention, dim = -1)
        
        out = torch.matmul(attention, g.permute(0, 2, 1)) # (B, THW, C // 2)
        out = out.permute(0, 2, 1).view(B, C // 2, T, H, W)
        
        out = self.conv_z(out)
        out = self.bn(out)
        
        out += residual
        out = self.relu(out)
    
class ResNet(nn.Module):
    
    def __init__(
        self,
        block, # BasicBlock or Bottleneck
        layers, # number of blocks for each layer
        block_inplanes, # number of input channels for each layer
        n_input_channels = 3, # number of input channels
        conv1_t_size = 7, # kernel size in t for the first conv layer
        conv1_t_stride = 1, # stride in t for the first conv layer
        no_max_pool = False, # whether to use max pool
        widen_factor = 1.0, # widen factor
        nl_nums = 0, # number of non_local block
        nl_subsample = False, # apply supsample for non_local block
        n_classes = 27 # number of classes
    ):
        super().__init__()
        
        self.nl_nums = nl_nums
        self.nl_subsample = nl_subsample
        
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        
        # First convolution
        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size = (conv1_t_size, 7, 7),
            stride = (conv1_t_stride, 2, 2),
            padding = (conv1_t_size // 2, 3, 3),
            bias = False
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool3d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )
        self.nl1 = NLBlock(self.in_planes, subsample = self.nl_subsample)
        
        # Layer 1
        self.layer1 = self._make_layer(
            block,
            block_inplanes[0],
            layers[0],
        )
        self.nl2 = NLBlock(self.in_planes, subsample = self.nl_subsample)
        # Layer 2
        self.layer2 = self._make_layer(
            block,
            block_inplanes[1],
            layers[1],
            stride = 2
        )
        self.nl3 = NLBlock(self.in_planes, subsample = self.nl_subsample)
        # Layer 3
        self.layer3 = self._make_layer(
            block,
            block_inplanes[2],
            layers[2],
            stride = 2
        )
        self.nl4 = NLBlock(self.in_planes, subsample = self.nl_subsample)
        # Layer 4
        self.layer4 = self._make_layer(
            block,
            block_inplanes[3],
            layers[3],
            stride = 2
        )
        self.nl5 = NLBlock(self.in_planes, subsample = self.nl_subsample)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode = 'fan_out',
                    nonlinearity = 'relu'
                )
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
         
    # Downsample input x and zero padding before adding it with out (BasicBlock and Bottleneck)    
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size = 1, stride = stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1),
            out.size(2), out.size(3), out.size(4)
        )
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        
        out = torch.cat([out.data, zero_pads], dim = 1)
        
        return out
    
    def _make_layer(self, block, planes, num_blocks, stride = 1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = partial(
                self._downsample_basic_block,
                planes = planes * block.expansion,
                stride = stride
            )
        
        layers = []
        layers.append(
            block(
                in_planes = self.in_planes,
                planes = planes,
                stride = stride,
                downsample = downsample
            )
        )
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        if self.nl_nums >= 2:
            x = self.nl1(x)
            
        x = self.layer1(x)
        if self.nl_nums >= 3:
            x = self.nl2(x)
        x = self.layer2(x)
        if self.nl_nums >= 1:
            x = self.nl3(x)
        x = self.layer3(x)
        if self.nl_nums >= 4:
            x = self.nl4(x)
        x = self.layer4(x)
        if self.nl_nums >= 5:
            x = self.nl5(x)
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
def R3D(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152]
    
    if model_depth == 10:
        model = ResNet(
            BasicBlock, # block
            [1, 1, 1, 1], # layers
            get_inplanes(), # block_inplanes: [64, 128, 256, 512]
            **kwargs # others
        )
    elif model_depth == 18:
        model = ResNet(
            BasicBlock,
            [2, 2, 2, 2],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 34:
        model = ResNet(
            BasicBlock,
            [3, 4, 6, 3],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 50:
        model = ResNet(
            Bottleneck,
            [3, 4, 6, 3],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 101:
        model = ResNet(
            Bottleneck,
            [3, 4, 23, 3],
            get_inplanes(),
            **kwargs
        )
    elif model_depth == 152:
        model = ResNet(
            Bottleneck,
            [3, 8, 36, 3],
            get_inplanes(),
            **kwargs
        )
    
    return model

def main():
    model = R3D(
        50,
        n_input_channels = 3,
        conv1_t_size = 7,
        conv1_t_stride = 1,
        no_max_pool = False,
        widen_factor = 1.0,
        n_classes = 27
    )

    summary(model, (3, 30, 112, 112))
    
if __name__ == '__main__':
    main()