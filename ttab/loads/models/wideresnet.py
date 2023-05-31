import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(
        self, depth, widen_factor, num_classes, split_point="layer3", dropout_rate=0.3
    ):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert split_point in ["layer2", "layer3", None], "invalid split position."
        self.split_point = split_point

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) / 6
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout_rate, stride=1
        )
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout_rate, stride=2
        )
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout_rate, stride=2
        )
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(nStages[3], num_classes, bias=False)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward_features(self, x):
        """Forward function without classifier. Use gradient checkpointing to save memory."""
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.split_point in ["layer3", None]:
            x = self.layer3(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        """Forward function for classifier. Use gridient checkpointing to save memory."""
        if self.split_point == "layer2":
            x = self.layer3(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
