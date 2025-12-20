"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# from torchvision.models import resnet18, resnet34, resnet50, resnet101             #


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class SupConResNet_MultiHead(nn.Module):
    """backbone + projection head"""

    def __init__(self, input_size=64, feat_dim=128, in_channels=3, zero_init_residual=False):
        super(SupConResNet_MultiHead, self).__init__()
        num_blocks = [2, 2, 2, 2]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=2)
        self.avgpool1 = nn.AdaptiveAvgPool2d((int(input_size/4), int(input_size/4)))
        self.avgpool2 = nn.AdaptiveAvgPool2d((int(input_size/8), int(input_size/8)))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.output_head1 = nn.Sequential(
            nn.Linear(128*int(input_size/4)*int(input_size/4),
                      64*int(input_size/4)*int(input_size/4)),
            nn.ReLU(inplace=True),
            nn.Linear(64*int(input_size/4)*int(input_size/4), feat_dim)
        )

        self.output_head2 = nn.Sequential(
            nn.Linear(256*int(input_size/8)*int(input_size/8),
                      128*int(input_size/8)*int(input_size/8)),
            nn.ReLU(inplace=True),
            nn.Linear(128*int(input_size/8)*int(input_size/8), feat_dim)
        )

        self.output_head3 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        feat1 = self.avgpool1(out)
        feat1 = torch.flatten(feat1, 1)
        feat1 = self.output_head1(feat1)
        feat1 = F.normalize(feat1, dim=1)

        out = self.layer3(out)
        feat2 = self.avgpool2(out)
        feat2 = torch.flatten(feat2, 1)
        feat2 = self.output_head2(feat2)
        feat2 = F.normalize(feat2, dim=1)

        out = self.layer4(out)
        feat3 = self.avgpool3(out)
        feat3 = torch.flatten(feat3, 1)
        feat3 = self.output_head3(feat3)
        feat3 = F.normalize(feat3, dim=1)

        return feat1, feat2, feat3



if __name__ == "__main__":

    model = SupConResNet_MultiHead()
    x = torch.ones(1, 3, 32, 32)
    f1, f2, f3 = model(x)