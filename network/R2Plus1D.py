import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
# from torchsummary import summary

def get_inplanes():
    return [64, 128, 256, 512]

# Spatial Convolution (2)
def conv1x3x3(in_planes, mid_planes, stride = 1):
    return nn.Conv3d(
        in_planes,
        mid_planes,
        kernel_size = (1, 3, 3),
        stride = (1, stride, stride),
        padding = (0, 1, 1),
        bias = False
    )

# Temporal Convolution (1)
def conv3x1x1(mid_planes, planes, stride = 1):
    return nn.Conv3d(
        mid_planes,
        planes,
        kernel_size = (3, 1, 1),
        stride = (stride, 1, 1),
        padding = (1, 0, 0),
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

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1 # Read the paper for more details (page 4/10)
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace = True)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

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

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block, # BasicBlock or Bottleneck
        layers, # number of blocks for each layer
        block_inplanes, # number of input channels for each layer
        n_input_channels = 3, # number of input channels
        conv1_t_size = 7, # kernel size in t dim for the first conv layer
        conv1_t_stride = 1, # stride in t dim for the first conv layer
        no_max_pool = False, # whether to use max pool
        widen_factor = 1.0, # widen factor
        n_classes = 28 # number of classes
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # First convolution
        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        # Spatial convolution
        self.conv1_s = nn.Conv3d(
            n_input_channels,
            mid_planes,
            kernel_size = (1, 7, 7),
            stride = (1, 2, 2),
            padding = (0, 3, 3),
            bias = False
        )
        self.bn1_s = nn.BatchNorm3d(mid_planes)
        # Temporal convolution
        self.conv1_t = nn.Conv3d(
            mid_planes,
            self.in_planes,
            kernel_size = (conv1_t_size, 1, 1),
            stride = (conv1_t_stride, 1, 1),
            padding = (conv1_t_size // 2, 0, 0),
            bias = False
        )
        self.bn1_t = nn.BatchNorm3d(self.in_planes)
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
        # Layer 2
        self.layer2 = self._make_layer(
            block,
            block_inplanes[1],
            layers[1],
            stride = 2
        )
        # Layer 3
        self.layer3 = self._make_layer(
            block,
            block_inplanes[2],
            layers[2],
            stride = 2
        )
        # Layer 4
        self.layer4 = self._make_layer(
            block,
            block_inplanes[3],
            layers[3],
            stride = 2
        )

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

    def _make_layer(self, block, planes, blocks, stride = 1):
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
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def R2Plus1D(model_depth, **kwargs):
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
    """
    model = R2Plus1D(
        18,
        n_input_channels = 3,
        conv1_t_size = 7,
        conv1_t_stride = 1,
        no_max_pool = True,
        widen_factor = 1.0,
        n_classes = 28
    )
    
    summary(model, (3, 30, 112, 112))
    """
    
if __name__ == '__main__':
    main()
    