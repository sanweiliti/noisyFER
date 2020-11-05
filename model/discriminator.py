import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)


class Attention(nn.Module):
    # input: ch * h * w
    # output: ch * h * w
    def __init__(self, ch, SN_eps=1e-12):
        super(Attention, self).__init__()
        # Channel multiplier
        self.SN_eps = SN_eps
        self.ch = ch
        self.theta = snconv2d(in_channels=self.ch, out_channels=self.ch // 8, kernel_size=1, padding=0,
                              bias=False, eps=self.SN_eps)
        self.phi = snconv2d(in_channels=self.ch, out_channels=self.ch // 8, kernel_size=1, padding=0, bias=False,
                            eps=self.SN_eps)
        self.g = snconv2d(in_channels=self.ch, out_channels=self.ch // 2, kernel_size=1, padding=0,
                          bias=False, eps=self.SN_eps)
        self.o = snconv2d(in_channels=self.ch // 2, out_channels=self.ch, kernel_size=1, padding=0,
                          bias=False, eps=self.SN_eps)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, SN_eps=1e-12, wide=True,
                 preactivation=False, downsample=True):
        super(DBlock, self).__init__()
        self.SN_eps = SN_eps
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels

        self.preactivation = preactivation
        self.activation = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.pooling = nn.AvgPool2d(2)

        # Conv layers
        self.conv1 = snconv2d(in_channels=self.in_channels, out_channels=self.hidden_channels,
                              kernel_size=3, padding=1, eps=self.SN_eps)
        self.conv2 = snconv2d(in_channels=self.hidden_channels, out_channels=self.out_channels,
                              kernel_size=3, padding=1, eps=self.SN_eps)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False  # use conv1*1 in shortcut or not
        if self.learnable_sc:
            self.conv_sc = snconv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=1, padding=0, eps=self.SN_eps)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv_sc(x)
        if self.downsample:
            x = self.pooling(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)  # This line *must* be an out-of-place ReLU or it will negatively affect the shortcut connection.
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.pooling(h)
        return h + self.shortcut(x)


class MLP_large(nn.Module):
    def __init__(self, in_channels, out_channels, SN_eps=1e-12):
        super(MLP_large, self).__init__()
        self.inchannels = in_channels
        self.outchannels = out_channels
        self.SN_eps = SN_eps
        self.linear = snlinear(in_features=self.inchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear1 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear2 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear3 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear4 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear5 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear6 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear7 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)
        self.linear8 = snlinear(in_features=self.outchannels, out_features=self.outchannels, eps=self.SN_eps)

    def forward(self, x):
        x = self.linear(x)
        x_0 = F.relu(x)
        x_1 = self.linear2(F.relu(self.linear1(x_0)))
        x_3 = self.linear4(F.relu(self.linear3(x_0)))
        x_5 = self.linear6(F.relu(self.linear5(x_0)))
        x_7 = self.linear8(F.relu(self.linear7(x_0)))
        out = x + x_1 + x_3 + x_5 + x_7
        return out


# image stream
class Discriminator_x(nn.Module):
    def __init__(self, img_size=256, out_channels=1024, SN_eps=1e-12, wide=True, with_sx=False, early_xfeat=False):
        super(Discriminator_x, self).__init__()
        self.SN_eps = SN_eps
        self.wide = wide
        self.pool = nn.AvgPool2d(2)
        self.img_size = img_size
        self.with_sx = with_sx
        self.early_xfeat = early_xfeat
        self.DBlock = DBlock(in_channels=3, out_channels=64, SN_eps=self.SN_eps, wide=self.wide,
                              preactivation=False, downsample=True)
        self.DBlock1 = DBlock(in_channels=64, out_channels=128, SN_eps=self.SN_eps, wide=self.wide,
                              preactivation=True, downsample=True)
        self.DBlock2 = DBlock(in_channels=128, out_channels=256, SN_eps=self.SN_eps, wide=self.wide,
                              preactivation=True, downsample=True)
        self.DBlock3 = DBlock(in_channels=256, out_channels=512, SN_eps=self.SN_eps, wide=self.wide,
                              preactivation=True, downsample=True)
        self.DBlock4 = DBlock(in_channels=512, out_channels=out_channels, SN_eps=self.SN_eps, wide=self.wide,
                              preactivation=True, downsample=True)
        self.DBlock5 = DBlock(in_channels=out_channels, out_channels=out_channels, SN_eps=self.SN_eps, wide=self.wide,
                              preactivation=True, downsample=False)
        self.attention = Attention(ch=64, SN_eps=self.SN_eps)

        self.score_x = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)  # todo: with/without spetral norm?

    def forward(self, x):
        # img stream: similiar with bigGAN discriminator
        if self.img_size > 32:
            x = self.pool(x)  # /2
        x_64 = self.DBlock(x)  # 64*(h/4)*(w/4)
        x_att = self.attention(x_64)  # 64*(h/4)*(w/4)
        x_32 = self.DBlock1(x_att)  # 128*(h/8)*(w/8)
        x_16 = self.DBlock2(x_32)   # 256*(h/16)*(w/16)
        x_8 = self.DBlock3(x_16)   # 512*(h/32)*(w/32)
        x_4 = self.DBlock4(x_8)    # out_channels*(h/64)*(w/64)
        x_4_2 = self.DBlock5(x_4)  # out_channels*(h/64)*(w/64)
        x_4_2 = F.relu(x_4_2)  # out_channels*(h/64)*(w/64)
        x_feat = torch.sum(torch.sum(x_4_2, 2), 2)  # bs*out_channels
        if self.with_sx and self.early_xfeat:
            s_x = self.score_x(x_feat)
            return x_feat, s_x, x_4
        elif self.with_sx:
            s_x = self.score_x(x_feat)
            return x_feat, s_x
        else:
            return x_feat


# label stream
class Discriminator_lbl(nn.Module):
    def __init__(self, in_channels=10, out_channels=512, SN_eps=1e-12):
        super(Discriminator_lbl, self).__init__()
        self.SN_eps = SN_eps
        self.z_mlp = MLP_large(in_channels=in_channels, out_channels=out_channels, SN_eps=self.SN_eps)
        self.score_z = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)

    def forward(self, z):
        # label stream: MLP
        z_feat = self.z_mlp(z)  # [bs, out_channels]
        s_z = self.score_z(z_feat)  # [bs, 1]
        return z_feat, s_z


# joint stream: MLP
class Discriminator_joint(nn.Module):
    def __init__(self, in_channels=2048, out_channels=1024, SN_eps=1e-12):
        super(Discriminator_joint, self).__init__()
        self.SN_eps = SN_eps
        self.in_channels = in_channels
        self.z_x_mlp = MLP_large(in_channels=in_channels, out_channels=out_channels, SN_eps=self.SN_eps)
        self.score_x_z = snlinear(in_features=out_channels, out_features=1, eps=self.SN_eps)

    def forward(self, x_feat, z_feat):
        # joint steam: MLP
        joint_feat = torch.cat([x_feat, z_feat], dim=1)  # [bs, 2048]
        res = self.z_x_mlp(joint_feat)  # [bs, 1024]
        s_x_z = self.score_x_z(res)
        return s_x_z


















