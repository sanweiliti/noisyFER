from model.discriminator import Discriminator_x, Discriminator_lbl, Discriminator_joint
from model.encoder_net import Encoder
from model.decoder_net import Decoder_64, Decoder_128
import torch
import torch.nn.functional as F
import itertools
from loss import *
from model.base_model import BaseModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiTaskModel(BaseModel):
    def __init__(self, args):
        super(MultiTaskModel, self).initialize(args)
        self.latent_dim = 7 + 2  # 7 emotion categories, 2 for valence/arousal valudes
        self.n_classes = 7
        self.encoder = Encoder(img_size=self.args.img_size, fc_layer=self.args.fc_layer,
                               latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)
        if self.args.img_size == 64:
            self.decoder = Decoder_64(img_size=self.args.img_size,
                                      latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)
        elif self.args.img_size == 128:
            self.decoder = Decoder_128(img_size=self.args.img_size,
                                       latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)

        if self.args.isTrain:
            self.train_model_name = ['encoder', 'decoder']
            self.train_model_name += ['discriminator_x', 'discriminator_z0',
                                      'discriminator_z1_exp', 'discriminator_z1_va', 'discriminator_joint_xz']

            self.discriminator_x = Discriminator_x(img_size=self.args.img_size, out_channels=512, wide=True,
                                                   with_sx=True, early_xfeat=False).to(device)
            self.discriminator_z0 = Discriminator_lbl(in_channels=self.args.noise_dim, out_channels=512).to(device)
            self.discriminator_z1_exp = Discriminator_lbl(in_channels=7, out_channels=512).to(device)
            self.discriminator_z1_va = Discriminator_lbl(in_channels=2, out_channels=512).to(device)

            self.discriminator_joint_xz = Discriminator_joint(in_channels=512*4).to(device)

            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                       itertools.chain(self.encoder.parameters(),
                                                                       self.decoder.parameters(),
                                                                       )),
                                                lr=self.args.lr)
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                       itertools.chain(self.discriminator_x.parameters(),
                                                                       self.discriminator_z0.parameters(),
                                                                       self.discriminator_z1_exp.parameters(),
                                                                       self.discriminator_z1_va.parameters(),
                                                                       self.discriminator_joint_xz.parameters(),
                                                                       )),
                                                lr=self.args.lr)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)


    def set_input(self, data):
        self.img = data[0].to(device)  # [batch_size, 3, 256, 256]
        self.exp_lbl = data[1].to(device)
        self.va_lbl = data[2].to(device)

        self.batch_size = self.img.shape[0]
        self.exp_lbl_onehot = torch.zeros(self.batch_size, self.n_classes).to(device)
        self.exp_lbl_onehot[range(self.batch_size), self.exp_lbl] = 1.0  # [bs, 7], float, require_grad=F


    def sample_z(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        return z_mu + z_std * eps

    def forward_G(self, epoch):
        # encoder
        mean, logvar, self.z0_enc, self.z1_enc = self.encoder(self.img)
        self.z1_exp_enc = self.z1_enc[:, 0:7]
        self.z1_va_enc = self.z1_enc[:, 7:]

        if epoch > self.args.gan_start_epoch:
            # decoder
            self.z0_dec = torch.randn(self.batch_size, self.args.noise_dim).to(device)
            self.z1_exp_dec = self.exp_lbl_onehot
            self.z1_va_dec = self.va_lbl

            self.dec_img = self.decoder(torch.cat([self.z0_dec, self.z1_exp_dec, self.z1_va_dec], dim=1))


    def backward_G(self, epoch):
        # exp loss + va loss
        self.loss_class = F.cross_entropy(input=self.z1_exp_enc, target=self.exp_lbl, reduction='mean')
        loss_v = ccc(self.z1_va_enc[:, 0], self.va_lbl[:, 0])
        loss_a = ccc(self.z1_va_enc[:, 1], self.va_lbl[:, 1])
        self.loss_va = 1 - (loss_v + loss_a) / 2

        if epoch > self.args.gan_start_epoch:
            # encoder gan loss
            # marginal scores
            x_feat_enc, s_x_enc = self.discriminator_x(self.img)
            z0_feat_enc, s_z0_enc = self.discriminator_z0(self.z0_enc)
            z1_exp_feat_enc, s_z1_exp_enc = self.discriminator_z1_exp(self.z1_exp_enc)
            z1_va_feat_enc, s_z1_va_enc = self.discriminator_z1_va(self.z1_va_enc)
            # joint score
            s_xz_enc = self.discriminator_joint_xz(x_feat_enc, torch.cat([z0_feat_enc, z1_exp_feat_enc, z1_va_feat_enc], dim=1))
            score_enc_xz = torch.cat([self.args.lambda_sx * s_x_enc,
                                      self.args.lambda_sz0 * s_z0_enc,
                                      self.args.lambda_sz1_exp * s_z1_exp_enc,
                                      self.args.lambda_sz1_va * s_z1_va_enc,
                                      self.args.lambda_sxz * s_xz_enc], 1)
            # decoder gan loss
            # marginal scores
            x_feat_dec, s_x_dec = self.discriminator_x(self.dec_img)
            z0_feat_dec, s_z0_dec = self.discriminator_z0(self.z0_dec)
            z1_exp_feat_dec, s_z1_exp_dec = self.discriminator_z1_exp(self.z1_exp_dec)
            z1_va_feat_dec, s_z1_va_dec = self.discriminator_z1_va(self.z1_va_dec)
            # joint score
            s_xz_dec = self.discriminator_joint_xz(x_feat_dec, torch.cat([z0_feat_dec, z1_exp_feat_dec, z1_va_feat_dec], dim=1))
            score_dec_xz = torch.cat([self.args.lambda_sx * s_x_dec,
                                      self.args.lambda_sz0 * s_z0_dec,
                                      self.args.lambda_sz1_exp * s_z1_exp_dec,
                                      self.args.lambda_sz1_va * s_z1_va_dec,
                                      self.args.lambda_sxz * s_xz_dec], 1)

            self.loss_gan = torch.mean(score_enc_xz) + torch.mean(-score_dec_xz)

            # self.G_s_x = torch.mean(s_z0_enc) + torch.mean(-s_z0_dec)
            # self.G_s_z0 = torch.mean(s_x_enc) + torch.mean(-s_x_dec)
            # self.G_s_z1_exp = torch.mean(s_z1_exp_enc) + torch.mean(-s_z1_exp_dec)
            # self.G_s_z1_va = torch.mean(s_z1_va_enc) + torch.mean(-s_z1_va_dec)
            # self.G_s_xz = torch.mean(s_xz_enc) + torch.mean(-s_xz_dec)

            self.loss_G = self.args.lambda_exp * self.loss_class + \
                          self.args.lambda_va * self.loss_va + \
                          self.args.lambda_gan * self.loss_gan
        else:
            self.loss_G = self.args.lambda_exp * self.loss_class + self.args.lambda_va * self.loss_va
        self.loss_G.backward()


    def backward_D(self):
        # marginal scores
        x_feat_enc, s_x_enc = self.discriminator_x(self.img)
        z0_feat_enc, s_z0_enc = self.discriminator_z0(self.z0_enc.detach())
        z1_exp_feat_enc, s_z1_exp_enc = self.discriminator_z1_exp(self.z1_exp_enc.detach())
        z1_va_feat_enc, s_z1_va_enc = self.discriminator_z1_va(self.z1_va_enc.detach())
        # joint score
        s_xz_enc = self.discriminator_joint_xz(x_feat_enc, torch.cat([z0_feat_enc, z1_exp_feat_enc, z1_va_feat_enc], dim=1))
        score_enc_xz = torch.cat([self.args.lambda_sx * s_x_enc,
                                  self.args.lambda_sz0 * s_z0_enc,
                                  self.args.lambda_sz1_exp * s_z1_exp_enc,
                                  self.args.lambda_sz1_va * s_z1_va_enc,
                                  self.args.lambda_sxz * s_xz_enc], 1)

        # marginal scores
        x_feat_dec, s_x_dec = self.discriminator_x(self.dec_img.detach())
        z0_feat_dec, s_z0_dec = self.discriminator_z0(self.z0_dec)
        z1_exp_feat_dec, s_z1_exp_dec = self.discriminator_z1_exp(self.z1_exp_dec)
        z1_va_feat_dec, s_z1_va_dec = self.discriminator_z1_va(self.z1_va_dec)
        # joint score
        s_xz_dec = self.discriminator_joint_xz(x_feat_dec, torch.cat([z0_feat_dec, z1_exp_feat_dec, z1_va_feat_dec], dim=1))
        score_dec_xz = torch.cat([self.args.lambda_sx * s_x_dec,
                                  self.args.lambda_sz0 * s_z0_dec,
                                  self.args.lambda_sz1_exp * s_z1_exp_dec,
                                  self.args.lambda_sz1_va * s_z1_va_dec,
                                  self.args.lambda_sxz * s_xz_dec], 1)

        loss_real_xz = torch.mean(F.relu(1. - score_enc_xz))
        loss_fake_xz = torch.mean(F.relu(1. + score_dec_xz))
        self.loss_D_gan = loss_real_xz + loss_fake_xz

        # self.D_s_x = torch.mean(F.relu(1. - s_x_enc)) + torch.mean(F.relu(1. + s_x_dec))
        # self.D_s_z0 = torch.mean(F.relu(1. - s_z0_enc)) + torch.mean(F.relu(1. + s_z0_dec))
        # self.D_s_z1_exp = torch.mean(F.relu(1. - s_z1_exp_enc)) + torch.mean(F.relu(1. + s_z1_exp_dec))
        # self.D_s_z1_va = torch.mean(F.relu(1. - s_z1_va_enc)) + torch.mean(F.relu(1. + s_z1_va_dec))
        # self.D_s_xz = torch.mean(F.relu(1. - s_xz_enc)) + torch.mean(F.relu(1. + s_xz_dec))

        self.loss_D = self.loss_D_gan
        self.loss_D.backward()


    def func_require_grad(self, model_, flag_):
        for mm in model_:
            self.set_requires_grad(mm, flag_)

    def func_zero_grad(self, model_):
        for mm in model_:
            mm.zero_grad()

    def optimize_params(self, epoch):
        self.encoder.train()
        self.decoder.train()
        self.forward_G(epoch)

        # D
        if epoch > self.args.gan_start_epoch:
            self.func_require_grad([self.discriminator_x,
                                    self.discriminator_z0, self.discriminator_z1_exp, self.discriminator_z1_va,
                                    self.discriminator_joint_xz], True)
            for i in range(self.args.iter_D):
                self.func_zero_grad([self.discriminator_x,
                                    self.discriminator_z0, self.discriminator_z1_exp, self.discriminator_z1_va,
                                    self.discriminator_joint_xz])
                self.backward_D()
                self.optimizer_D.step()

        # G
        self.func_require_grad([self.discriminator_x,
                                self.discriminator_z0, self.discriminator_z1_exp, self.discriminator_z1_va,
                                self.discriminator_joint_xz], False)
        for i in range(self.args.iter_G):
            self.optimizer_G.zero_grad()
            if i > 0:
                self.forward_G(epoch)
            self.backward_G(epoch)
            self.optimizer_G.step()

