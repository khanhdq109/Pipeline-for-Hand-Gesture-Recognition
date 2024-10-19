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
    
    def __init__(self, in_channels, subsample = True):
        super(NLBlock, self).__init__()
        
        self.n = 2
        
        self.in_channels = in_channels
        self.inter_channels = in_channels // self.n
        
        # Theta, Phi, G transforms
        self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size = 1)
        self.phi = nn.Conv3d(in_channels, self.inter_channels, kernel_size = 1)
        self.g = nn.Conv3d(in_channels, self.inter_channels, kernel_size = 1)
        
        self.bn1 = nn.BatchNorm3d(self.inter_channels)
        
        # W transform to match input channel dimensions
        self.W = nn.Conv3d(self.inter_channels, in_channels, kernel_size = 1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        
        self.relu = nn.ReLU(inplace = True)
        self.softmax = nn.Softmax(dim = -1)
        
        self.subsample = subsample
        if self.subsample:
            self.pool = nn.MaxPool3d(kernel_size = 2)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Theta, Phi and G projections
        theta_x = self.relu(self.bn1(self.theta(x)))
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)
        
        phi_x = self.relu(self.bn1(self.phi(x)))
        if self.subsample:
            phi_x = self.pool(phi_x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        
        g_x = self.relu(self.bn1(self.g(x)))
        if self.subsample:
            g_x = self.pool(g_x)
        g_x = g_x.view(batch_size, self.inter_channels, -1)
        
        # Compute attention map
        theta_x = theta_x.permute(0, 2, 1) # (B, THW, C)
        f = torch.matmul(theta_x, phi_x) # (B, THW, C) x (B, C, THW)
        f_div_C = self.softmax(f) # (B, THW, THW)
        
        # Apply attention to G
        y = torch.matmul(f_div_C, g_x.permute(0, 2, 1)) # (B, THW, THW) x (B, THW, C) --> (B, THW, C)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, self.inter_channels, *x.size()[2:]) 
        
        # Apply the final W transform
        W_y = self.W(y)
        W_y = self.bn2(W_y)
        z = W_y + x # Residual connection
        z = self.relu(z)
        
        return z
    
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
        nl_nums = 0, # number of non-local block
        n_classes = 27 # number of classes
    ):
        super().__init__()
        
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.nl_nums = nl_nums
        
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
        
        # Layer 1
        self.layer1 = self._make_layer(
            block,
            block_inplanes[0],
            layers[0],
        )
        if self.nl_nums >= 1: self.nl1 = NLBlock(block_inplanes[0])
        # Layer 2
        self.layer2 = self._make_layer(
            block,
            block_inplanes[1],
            layers[1],
            stride = 2
        )
        if self.nl_nums >= 2: self.nl2 = NLBlock(block_inplanes[1])
        # Layer 3
        self.layer3 = self._make_layer(
            block,
            block_inplanes[2],
            layers[2],
            stride = 2
        )
        if self.nl_nums >= 3: self.nl3 = NLBlock(block_inplanes[2])
        # Layer 4
        self.layer4 = self._make_layer(
            block,
            block_inplanes[3],
            layers[3],
            stride = 2
        )
        if self.nl_nums >= 4: self.nl4 = NLBlock(block_inplanes[3])
        
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
            out.size(2), out.size(3), out.size(4),
            device = out.device
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
        
        # Layer 1
        x = self.layer1(x)
        if self.nl_nums >= 1:
            x = self.nl1(x)
        # Layer 2
        x = self.layer2(x)
        if self.nl_nums >= 2:
            x = self.nl2(x)
        # Layer 3
        x = self.layer3(x)
        if self.nl_nums >= 3:
            x = self.nl3(x)
        # Layer 4
        x = self.layer4(x)
        if self.nl_nums >= 4:
            x = self.nl4(x)
        
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = R3D(
        34,
        n_input_channels = 3,
        conv1_t_size = 7,
        conv1_t_stride = 1,
        no_max_pool = True,
        widen_factor = 1.0,
        nl_nums = 1,
        n_classes = 27
    ).to(device)

    summary(model, (3, 30, 112, 112), device = str(device))
    
if __name__ == '__main__':
    main()