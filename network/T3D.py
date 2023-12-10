import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

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

class DenseLayer(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, growth_rate, stride = 1, dropout = 0.0):
        super().__init__()
        
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = conv1x1x1(in_planes, self.expansion * growth_rate, stride)
        
        self.bn2 = nn.BatchNorm3d(self.expansion * growth_rate)
        self.conv2 = conv3x3x3(self.expansion * growth_rate, growth_rate)
        
        self.dropout = dropout
        
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        if self.dropout > 0:
            out = F.dropout(out, p = self.dropout, training = self.training)
        
        return torch.cat([x, out], 1)
    
class TransitionLayer(nn.Module):
    
    def __init__(self, in_planes, stride = 1, phi = 0.5, dropout = 0.0):
        super().__init__()
        
        self.bn = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv = conv1x1x1(in_planes, int(in_planes * phi), stride)
        
        self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2)
        
        self.dropout = dropout
        
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        
        if self.dropout > 0:
            out = F.dropout(out, p = self.dropout, training = self.training)
        
        out = self.pool(out)
        
        return out
    
class TemporalTransitionLayer(nn.Module):
    pass

class DenseNet(nn.Module):
    
    def __init__(
        self,
        transition, # TransitionLayer or TemporalTransitionLayer
        layers, # number of layers for each dense block
        phi = 0.5, # compression factor
        growth_rate = 12, # growth rate
        n_input_channels = 3, # number of input channels
        conv1_t_size = 7, # kernel size in t for the first conv layer
        conv1_t_stride = 1, # stride in t for the first conv layer
        no_max_pool = False, # whether to use max pool
        n_classes = 27, # number of classes
        dropout = 0.0, # dropout rate
    ):
        super().__init__()
        
        self.phi = phi
        self.growth_rate = growth_rate
        
        self.in_planes = 2 * growth_rate
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
        self.maxpool = nn.MaxPool3d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )
        
        # Block 1
        self.block1 = self._make_dense_block(
            layers[0],
            dropout = dropout,
            is_first = True
        )
        self.trans1 = transition(
            self.in_planes,
            phi = phi,
            dropout = dropout
        )
        
        # Block 2
        self.block2 = self._make_dense_block(
            layers[1],
            dropout = dropout,
            is_first = False
        )
        self.trans2 = transition(
            self.in_planes,
            phi = phi,
            dropout = dropout
        )
        
        # Block 3
        self.block3 = self._make_dense_block(
            layers[2],
            dropout = dropout,
            is_first = False
        )
        self.trans3 = transition(
            self.in_planes,
            phi = phi,
            dropout = dropout
        )
        
        # Block 4
        self.block4 = self._make_dense_block(
            layers[3],
            dropout = dropout,
            is_first = False
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self.in_planes, n_classes)
        
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
                
    def _make_dense_block(self, layer, stride = 1, dropout = 0.0, is_first = False):
        if not is_first:
            self.in_planes = int(self.in_planes * self.phi)
        layers = []
        for i in range(layer):
            layers.append(
                DenseLayer(
                    self.in_planes,
                    self.growth_rate,
                    stride,
                    dropout
                )
            )
            self.in_planes += self.growth_rate
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
            
        x = self.block1(x)
        x = self.trans1(x)
        
        x = self.block2(x)
        x = self.trans2(x)
        
        x = self.block3(x)
        x = self.trans3(x)
        
        x = self.block4(x)
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)   
        x = self.fc(x)
        
        return x
    
def D3D(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]
    
    if model_depth == 121:
        model = DenseNet(
            TransitionLayer,
            [6, 12, 24, 16],
            **kwargs
        )
    elif model_depth == 169:
        model = DenseNet(
            TransitionLayer,
            [6, 12, 32, 32],
            **kwargs
        )
    elif model_depth == 201:
        model = DenseNet(
            TransitionLayer,
            [6, 12, 48, 32],
            **kwargs
        )
    elif model_depth == 264:
        model = DenseNet(
            TransitionLayer,
            [6, 12, 64, 48],
            **kwargs
        )
        
    return model

def T3D(model_depth, **kwargs):
    assert model_depth in [121, 169, 201, 264]
    
    if model_depth == 121:
        model = DenseNet(
            TemporalTransitionLayer,
            [6, 12, 24, 16],
            **kwargs
        )
    elif model_depth == 169:
        model = DenseNet(
            TemporalTransitionLayer,
            [6, 12, 32, 32],
            **kwargs
        )
    elif model_depth == 201:
        model = DenseNet(
            TemporalTransitionLayer,
            [6, 12, 48, 32],
            **kwargs
        )
    elif model_depth == 264:
        model = DenseNet(
            TemporalTransitionLayer,
            [6, 12, 64, 48],
            **kwargs
        )
    
    return model

def main():
    model = D3D(
        121,
        phi = 0.5,
        growth_rate = 12,
        n_input_channels = 3,
        conv1_t_size = 7,
        conv1_t_stride = 1,
        no_max_pool = True,
        n_classes = 27,
        dropout = 0.0
    )
    
    # print(model)
    summary(model, (3, 30, 112, 112))
    
if __name__ == '__main__':
    main()