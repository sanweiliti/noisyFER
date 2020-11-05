import functools
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import math

class Vgg(nn.Module):
    def __init__(self, img_size=256, fc_layer=4096, classes=10):
        super(Vgg, self).__init__()
        self.fc_layer = fc_layer
        self.classes = classes
        if img_size == 256:
            self.final_size = 8
        if img_size == 96:
            self.final_size = 3
        if img_size == 64:
            self.final_size = 2
        if img_size == 32:
            self.final_size = 1

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # /2

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # /4

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # /8

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # /16

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )  # /32

        self.classifier = nn.Sequential(
            nn.Linear(512 * self.final_size * self.final_size, self.fc_layer),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(self.fc_layer, self.fc_layer),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(self.fc_layer, self.classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        conv5_flatten = conv5.view(conv5.shape[0], -1)  # [batch_size, c, h, w] --> [batch_size, c*h*w]

        score = self.classifier(conv5_flatten)  # [batch_size, 2]
        return score

    # copy all cony layer params from pretrained vggface (fc layers have different dims)
    def init_vggface_params(self, pretrained_model_path):
        pretrained_dict = torch.load(pretrained_model_path)
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k[0:4] == 'conv':
                conv_blk_id = k[4]
                conv_layer_id = k[6]
                new_state_name = 'conv_block' + conv_blk_id + '.' + str((int(conv_layer_id)-1)*2) + k[7:]
                new_state_dict[new_state_name] = v
        return new_state_dict
