from model.discriminator import Discriminator_x, Discriminator_lbl, Discriminator_joint
from model.encoder_net import Encoder
from model.decoder_net import Decoder
import torch
import torch.nn.functional as F
import itertools
from loss import *
from model.base_model import BaseModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the final model, different marginal scores for each label set
class InconsistLabelModel(BaseModel):
    def __init__(self, args):
        super(InconsistLabelModel, self).initialize(args)
        self.latent_dim = 10 * 3
        self.encoder = Encoder(img_size=self.args.img_size, fc_layer=self.args.fc_layer,
                               latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)
        self.decoder = Decoder(latent_dim=self.latent_dim, noise_dim=self.args.noise_dim).to(device)

        if self.args.isTrain:
            self.train_model_name = ['encoder', 'decoder']
            self.train_model_name += ['discriminator_x', 'discriminator_z0', 'discriminator_z1', 'discriminator_joint_xz']

            self.discriminator_x = Discriminator_x(img_size=self.args.img_size, out_channels=512, wide=True,
                                                   with_sx=True, early_xfeat=False).to(device)
            self.discriminator_z0 = Discriminator_lbl(in_channels=self.args.noise_dim, out_channels=512).to(device)
            self.discriminator_z1_1 = Discriminator_lbl(in_channels=10, out_channels=512).to(device)
            self.discriminator_z1_2 = Discriminator_lbl(in_channels=10, out_channels=512).to(device)
            self.discriminator_z1_3 = Discriminator_lbl(in_channels=10, out_channels=512).to(device)
            self.discriminator_joint_xz = Discriminator_joint(in_channels=512+512+512*3).to(device)

            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                       itertools.chain(self.encoder.parameters(),
                                                                       self.decoder.parameters(),
                                                                       )),
                                                lr=self.args.lr)
            self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                       itertools.chain(self.discriminator_x.parameters(),
                                                                       self.discriminator_z0.parameters(),
                                                                       self.discriminator_z1_1.parameters(),
                                                                       self.discriminator_z1_2.parameters(),
                                                                       self.discriminator_z1_3.parameters(),
                                                                       self.discriminator_joint_xz.parameters(),
                                                                       )),
                                                lr=self.args.lr)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)


    def set_input(self, data):
        self.img = data[0].to(device)  # [batch_size, 3, 256, 256]
        self.noise_lbl_set = data[1]
        self.noise_lbl_1, self.noise_lbl_2, self.noise_lbl_3 = [item.to(device) for item in self.noise_lbl_set]
        self.clean_lbl = data[2].to(device)  # [bs]
        self.major_lbl = data[3].to(device)
        self.n_classes = 10
        self.batch_size = self.img.shape[0]

        self.y1_onehot = torch.zeros(self.batch_size, self.n_classes).to(device)
        self.y1_onehot[range(self.batch_size), self.noise_lbl_1] = 1.0  # [bs, 10], float, require_grad=F

        self.y2_onehot = torch.zeros(self.batch_size, self.n_classes).to(device)
        self.y2_onehot[range(self.batch_size), self.noise_lbl_2] = 1.0  # [bs, 10], float, require_grad=F

        self.y3_onehot = torch.zeros(self.batch_size, self.n_classes).to(device)
        self.y3_onehot[range(self.batch_size), self.noise_lbl_3] = 1.0  # [bs, 10], float, require_grad=F


    def sample_z(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_std)
        return z_mu + z_std * eps

    def forward_G(self):
        # encoder
        mean, logvar, self.z0_enc, self.z1_enc = self.encoder(self.img)  # z0: noise vector, z1_enc: predicted labels (dim=30)
        self.y1_enc, self.y2_enc, self.y3_enc = self.z1_enc[:, 0:10], self.z1_enc[:, 10:20], self.z1_enc[:, 20:]

        # decoder
        self.z0_dec = torch.randn(self.batch_size, self.args.noise_dim).to(device)
        self.z1_dec = torch.cat([self.y1_onehot, self.y2_onehot, self.y3_onehot], dim=1)
        self.dec_img = self.decoder(torch.cat([self.z0_dec, self.z1_dec], dim=1))


    def backward_G(self, epoch):
        self.loss_class = F.cross_entropy(input=self.y1_enc, target=self.noise_lbl_1, reduction='mean') + \
                          F.cross_entropy(input=self.y2_enc, target=self.noise_lbl_2, reduction='mean') + \
                          F.cross_entropy(input=self.y3_enc, target=self.noise_lbl_3, reduction='mean')

        # encoder gan loss
        # marginal scores
        x_feat_enc, s_x_enc = self.discriminator_x(self.img)
        z0_feat_enc, s_z0_enc = self.discriminator_z0(self.z0_enc)
        z1_feat_enc_1, s_z1_enc_1 = self.discriminator_z1_1(self.y1_enc)
        z1_feat_enc_2, s_z1_enc_2 = self.discriminator_z1_2(self.y2_enc)
        z1_feat_enc_3, s_z1_enc_3 = self.discriminator_z1_3(self.y3_enc)
        # joint score
        s_xz_enc = self.discriminator_joint_xz(x_feat_enc,
                                               torch.cat([z0_feat_enc, z1_feat_enc_1, z1_feat_enc_2, z1_feat_enc_3], dim=1))
        score_enc_xz = torch.cat([self.args.lambda_sx * s_x_enc,
                                  self.args.lambda_sz0 * s_z0_enc,
                                  self.args.lambda_sz1 * s_z1_enc_1,
                                  self.args.lambda_sz1 * s_z1_enc_2,
                                  self.args.lambda_sz1 * s_z1_enc_3,
                                  self.args.lambda_sxz * s_xz_enc], 1)

        # decoder gan loss
        # marginal scores
        x_feat_dec, s_x_dec = self.discriminator_x(self.dec_img)
        z0_feat_dec, s_z0_dec = self.discriminator_z0(self.z0_dec)
        z1_feat_dec_1, s_z1_dec_1 = self.discriminator_z1_1(self.y1_onehot)
        z1_feat_dec_2, s_z1_dec_2 = self.discriminator_z1_2(self.y2_onehot)
        z1_feat_dec_3, s_z1_dec_3 = self.discriminator_z1_3(self.y3_onehot)
        # joint score
        s_xz_dec = self.discriminator_joint_xz(x_feat_dec,
                                               torch.cat([z0_feat_dec, z1_feat_dec_1, z1_feat_dec_2, z1_feat_dec_3], dim=1))
        score_dec_xz = torch.cat([self.args.lambda_sx * s_x_dec,
                                  self.args.lambda_sz0 * s_z0_dec,
                                  self.args.lambda_sz1 * s_z1_dec_1,
                                  self.args.lambda_sz1 * s_z1_dec_2,
                                  self.args.lambda_sz1 * s_z1_dec_3,
                                  self.args.lambda_sxz * s_xz_dec], 1)
        self.loss_gan = torch.mean(score_enc_xz) + torch.mean(-score_dec_xz)

        if epoch > self.args.gan_start_epoch:
            self.loss_G = self.args.lambda_class * self.loss_class + \
                          self.args.lambda_gan * self.loss_gan
        else:
            self.loss_G = self.args.lambda_class * self.loss_class
        self.loss_G.backward()


    def backward_D(self):
        # marginal scores
        x_feat_enc, s_x_enc = self.discriminator_x(self.img)
        z0_feat_enc, s_z0_enc = self.discriminator_z0(self.z0_enc.detach())
        z1_feat_enc_1, s_z1_enc_1 = self.discriminator_z1_1(self.y1_enc.detach())
        z1_feat_enc_2, s_z1_enc_2 = self.discriminator_z1_2(self.y2_enc.detach())
        z1_feat_enc_3, s_z1_enc_3 = self.discriminator_z1_3(self.y3_enc.detach())
        # joint score
        s_xz_enc = self.discriminator_joint_xz(x_feat_enc,
                                               torch.cat([z0_feat_enc, z1_feat_enc_1, z1_feat_enc_2, z1_feat_enc_3],
                                                         dim=1))
        score_enc_xz = torch.cat([self.args.lambda_sx * s_x_enc,
                                  self.args.lambda_sz0 * s_z0_enc,
                                  self.args.lambda_sz1 * s_z1_enc_1,
                                  self.args.lambda_sz1 * s_z1_enc_2,
                                  self.args.lambda_sz1 * s_z1_enc_3,
                                  self.args.lambda_sxz * s_xz_enc], 1)

        # marginal scores
        x_feat_dec, s_x_dec = self.discriminator_x(self.dec_img.detach())
        z0_feat_dec, s_z0_dec = self.discriminator_z0(self.z0_dec)
        z1_feat_dec_1, s_z1_dec_1 = self.discriminator_z1_1(self.y1_onehot)
        z1_feat_dec_2, s_z1_dec_2 = self.discriminator_z1_2(self.y2_onehot)
        z1_feat_dec_3, s_z1_dec_3 = self.discriminator_z1_3(self.y3_onehot)
        # joint score
        s_xz_dec = self.discriminator_joint_xz(x_feat_dec,
                                               torch.cat([z0_feat_dec, z1_feat_dec_1, z1_feat_dec_2, z1_feat_dec_3],
                                                         dim=1))
        score_dec_xz = torch.cat([self.args.lambda_sx * s_x_dec,
                                  self.args.lambda_sz0 * s_z0_dec,
                                  self.args.lambda_sz1 * s_z1_dec_1,
                                  self.args.lambda_sz1 * s_z1_dec_2,
                                  self.args.lambda_sz1 * s_z1_dec_3,
                                  self.args.lambda_sxz * s_xz_dec], 1)

        loss_real_xz = torch.mean(F.relu(1. - score_enc_xz))
        loss_fake_xz = torch.mean(F.relu(1. + score_dec_xz))
        self.loss_D = loss_real_xz + loss_fake_xz

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
        self.forward_G()

        # D
        if epoch > self.args.gan_start_epoch:
            self.func_require_grad([self.discriminator_x, self.discriminator_z0,
                                    self.discriminator_z1_1, self.discriminator_z1_2, self.discriminator_z1_3,
                                    self.discriminator_joint_xz], True)
            for i in range(self.args.iter_D):
                self.func_zero_grad([self.discriminator_x, self.discriminator_z0,
                                     self.discriminator_z1_1, self.discriminator_z1_2, self.discriminator_z1_3,
                                     self.discriminator_joint_xz])
                self.backward_D()
                self.optimizer_D.step()

        # G
        self.func_require_grad([self.discriminator_x, self.discriminator_z0,
                                self.discriminator_z1_1, self.discriminator_z1_2, self.discriminator_z1_3,
                                self.discriminator_joint_xz], False)
        for i in range(self.args.iter_G):
            self.optimizer_G.zero_grad()
            if i > 0:
                self.forward_G()
            self.backward_G(epoch)
            self.optimizer_G.step()

