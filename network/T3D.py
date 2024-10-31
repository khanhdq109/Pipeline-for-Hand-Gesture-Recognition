import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary

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
    
def downsample(x, x1, x2, x3):
    """ 
        Crop x1, x2, x3 to the shape of x along the temporal (2) dimension
    """
    # Compare shapes and crop individually
    crop_size1 = max(x1.size(2) - x.size(2), 0)
    crop_size2 = max(x2.size(2) - x.size(2), 0)
    crop_size3 = max(x3.size(2) - x.size(2), 0)
    
    if crop_size1 > 0:
        x1 = x1[:, :, :-crop_size1, :, :]
    if crop_size2 > 0:
        x2 = x2[:, :, :-crop_size2, :, :]
    if crop_size3 > 0:
        x3 = x3[:, :, :-crop_size3, :, :]
    
    return x1, x2, x3

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
        
        return torch.cat((x, out), 1)
    
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
    
    def __init__(
        self, in_planes, t_size = [1, 3, 4], growth_rate = 12, expansion = 1,
        stride = 1, phi = 0.5, dropout = 0.0
    ):
        super().__init__()
        
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv3d(
            in_planes,
            growth_rate * expansion,
            kernel_size = (t_size[0], 1, 1),
            stride = 1,
            padding = (t_size[0] // 2, 0, 0),
            bias = False
        )
        self.conv2 = nn.Conv3d(
            in_planes,
            growth_rate * expansion,
            kernel_size = (t_size[1], 3, 3),
            stride = 1,
            padding = (t_size[1] // 2, 1, 1),
            bias = False
        )
        self.conv3 = nn.Conv3d(
            in_planes,
            growth_rate * expansion,
            kernel_size = (t_size[2], 3, 3),
            stride = 1,
            padding = (t_size[2] // 2, 1, 1),
            bias = False
        )
        
        self.bn2 = nn.BatchNorm3d(growth_rate * expansion * 3)
        self.conv4 = nn.Conv3d(
            growth_rate * expansion * 3,
            int(in_planes * phi),
            kernel_size = 1,
            stride = stride,
            bias = False
        )
        
        self.pool = nn.AvgPool3d(kernel_size = 2, stride = 2)
        
        self.dropout = dropout
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        
        out1 = self.conv1(out)
        out2 = self.conv2(out)
        out3 = self.conv3(out)
        
        out1, out2, out3 = downsample(out, out1, out2, out3)
        
        out = torch.cat((out1, out2, out3), 1)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv4(out)
        
        if self.dropout > 0:
            out = F.dropout(out, p = self.dropout, training = self.training)
        
        out = self.pool(out)
        
        return out
    
class NLBlock(nn.Module):
    
    def __init__(self, in_channels, subsample = True):
        super(NLBlock, self).__init__()
        
        self.inter_channels = max(in_channels // 2, 1)  # Prevent zero channels
        
        # Batch norm
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(self.inter_channels)
        
        # Theta, Phi, G transforms
        self.theta = nn.Conv3d(in_channels, self.inter_channels, kernel_size = 1)
        self.phi = nn.Conv3d(in_channels, self.inter_channels, kernel_size = 1)
        self.g = nn.Conv3d(in_channels, self.inter_channels, kernel_size = 1)
        
        # W transform to match input channel dimensions
        self.W = nn.Conv3d(self.inter_channels, in_channels, kernel_size = 1)
        
        # Initialize W transform weights to zero
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        
        # Additional layers
        self.relu = nn.ReLU(inplace = True)
        self.softmax = nn.Softmax(dim = -1)
        self.subsample = subsample
        if self.subsample:
            self.pool = nn.MaxPool3d(kernel_size = 2)
        
    def forward(self, x):
        batch_size, _, t, h, w = x.size()
        
        # Theta, Phi and G projections
        theta_x = self.theta(self.relu(self.bn1(x))).view(batch_size, self.inter_channels, -1)
        
        phi_x = self.phi(self.relu(self.bn1(x)))
        if self.subsample and t > 1 and h > 1 and w > 1:
            phi_x = self.pool(phi_x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        
        g_x = self.g(self.relu(self.bn1(x)))
        if self.subsample and t > 1 and h > 1 and w > 1:
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
        W_y = self.relu(self.bn2(y))
        W_y = self.W(W_y)
        z = W_y + x # Residual connection
        
        return z

class DenseNet(nn.Module):
    
    def __init__(
        self,
        transition, # TransitionLayer or TemporalTransitionLayer
        layers, # number of layers for each dense block
        phi = 0.5, # compression factor
        growth_rate = 12, # growth rate
        temporal_expansion = 1, # temporal expansion factor
        transition_t1_size = [1, 3, 6], # kernel size in t for the first transition conv
        transition_t_size = [1, 3, 4], # kernel size in t for the other transition conv
        n_input_channels = 3, # number of input channels
        conv1_t_size = 7, # kernel size in t for the first conv layer
        conv1_t_stride = 1, # stride in t for the first conv layer
        no_max_pool = False, # whether to use max pool
        nl_nums = 0, # number of non-local block,
        nl_subsample = True, # apply subsample to non-local blocks
        n_classes = 27, # number of classes
        dropout = 0.0, # dropout rate
    ):
        super().__init__()
        
        self.phi = phi
        self.growth_rate = growth_rate
        self.nl_nums = nl_nums
        
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
            stride = 1,
            padding = 1
        )
        if self.nl_nums >= 1: self.nl0 = NLBlock(self.in_planes, nl_subsample)
        
        # Block 1
        self.block1 = self._make_dense_block(
            layers[0],
            dropout = dropout,
            is_first = True
        )
        if self.nl_nums >= 2: self.nl1 = NLBlock(self.in_planes, nl_subsample)
        if isinstance(transition, TemporalTransitionLayer):
            self.trans1 = transition(
                self.in_planes,
                t_size = transition_t1_size,
                growth_rate = growth_rate,
                expansion = temporal_expansion,
                phi = phi,
                dropout = dropout
            )
        else:
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
        if self.nl_nums >= 3: self.nl2 = NLBlock(self.in_planes, nl_subsample)
        if isinstance(transition, TemporalTransitionLayer):
            self.trans2 = transition(
                self.in_planes,
                t_size = transition_t_size,
                growth_rate = growth_rate,
                expansion = temporal_expansion,
                phi = phi,
                dropout = dropout
            )
        else: 
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
        if self.nl_nums >= 4: self.nl3 = NLBlock(self.in_planes, nl_subsample)
        if isinstance(transition, TemporalTransitionLayer):
            self.trans3 = transition(
                self.in_planes,
                t_size = transition_t_size,
                growth_rate = growth_rate,
                expansion = temporal_expansion,
                phi = phi,
                dropout = dropout
            )
        else:
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
        if self.nl_nums >= 5: self.nl4 = NLBlock(self.in_planes, nl_subsample)
        
        # Final batch norm
        self.bn = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace = True)
        
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
        if self.nl_nums >= 1:
            x = self.nl0(x)
            
        x = self.block1(x)
        if self.nl_nums >= 2:
            x = self.nl1(x)
        x = self.trans1(x)
        
        x = self.block2(x)
        if self.nl_nums >= 3:
            x = self.nl2(x)
        x = self.trans2(x)
        
        x = self.block3(x)
        if self.nl_nums >= 4:
            x = self.nl3(x)
        x = self.trans3(x)
        
        x = self.block4(x)
        if self.nl_nums >= 5:
            x = self.nl4(x)
        
        x = self.bn(x)
        x = self.relu(x)
        
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
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = T3D(
        121,
        phi = 0.5,
        growth_rate = 12,
        temporal_expansion = 1,
        transition_t1_size = [1, 3, 6],
        transition_t_size = [1, 3, 4],
        n_input_channels = 3,
        conv1_t_size = 3,
        conv1_t_stride = 1,
        no_max_pool = True,
        nl_nums = 3,
        n_classes = 27,
        dropout = 0.0
    ).to(device)
    
    summary(model, (3, 24, 112, 112))
    """
    
if __name__ == '__main__':
    main()