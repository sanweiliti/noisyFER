import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel():

    # modify parser to add command line options,
    # and also change the default values if needed
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def name(self):
        return 'BaseModel'

    def initialize(self, args):
        self.args = args
        self.isTrain = args.isTrain
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_model_name = []

    def set_input(self, input):
        pass

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self):
        self.print_networks(verbose=False)

    def load_networks_all(self, prefix):
        for name in self.train_model_name:
            if 'netD' in name:
                continue
            net = getattr(self, name)
            load_filename = '{}_{}.pth'.format(prefix, name)

            self.load_networks(net, load_filename)

    # load model
    def load_networks(self, model, path):
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        pretrainDict = torch.load(path, map_location=self.device)
        modelDict = model.state_dict()
        for kk, vv in pretrainDict.items():
            kk = kk.replace('module.', '')
            if kk in modelDict:
                modelDict[kk].copy_(vv)
            else:
                print('{} not in modelDict'.format(kk))
        # model.load_state_dict(pretrainDict)
        # print(modelDict.keys())

    # make models eval mode during test time
    def eval(self):
        for name in self.train_model_name:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self):
        with torch.no_grad():
            self.forward()

    # # get image paths
    # def get_image_paths(self):
    #     return self.image_paths

    def optimize_parameters(self):
        pass


    # save models to the disk
    def save_networks(self, epoch, logdir):
        for name in self.train_model_name:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(logdir, save_filename)
                net = getattr(self, name)
                if torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net = net.to(device)  # make sure net is on GPU
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # print network information

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.train_model_name:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' %
                      (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
