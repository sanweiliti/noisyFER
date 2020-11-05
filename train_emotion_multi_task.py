import torch
import time
from tqdm import tqdm
import argparse
from loader.dataloader_raf_multi_task import DataloaderRAF_MultiTask
from loader.dataloader_raf import DataloaderRAF
from loader.dataloader_affectnet_multi_task import DataloaderAffectnet_MultiTask
from utils import *
from torch.utils import data
from metrics import *
from model.multi_task_model import MultiTaskModel


####### training settings #########
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--img_size", default=64, type=int)
parser.add_argument("--log_step", default=100, type=int)
parser.add_argument("--val_step", default=100, type=int)
parser.add_argument("--save_step", default=200, type=int)
parser.add_argument("--num_epoch", default=100000, type=int)
parser.add_argument("--isTrain", default=True, type=bool)
parser.add_argument("--fc_layer", default=500, type=int)
parser.add_argument("--noise_dim", default=100, type=int, help='dimension of noise vector')

parser.add_argument("--base_dataset", default='raf', type=str, help='base dataset, raf-case/affectnet-case')  # raf/affectnet
parser.add_argument("--gan_start_epoch", default=5, type=int, help='start gan loss from which epoch')  # raf: 5 / affectnet: -1

parser.add_argument("--lambda_exp", default=1.0, type=float)
parser.add_argument("--lambda_va", default=1.0, type=float)
parser.add_argument("--lambda_gan", default=1.0, type=float)

# weights for marginal scores in D
parser.add_argument("--lambda_sx", default=1.0, type=float, help='loss weight for marginal score of image')
parser.add_argument("--lambda_sz0", default=1.0, type=float, help='loss weight for mariginal score of noise vector')
parser.add_argument("--lambda_sz1_exp", default=1.0, type=float,
                    help='loss weight for mariginal score of noisy discrete emotion label')
parser.add_argument("--lambda_sz1_va", default=1.0, type=float,
                    help='loss weight for mariginal score of noisy continuous va label')
# weights for joint score in D
parser.add_argument("--lambda_sxz", default=1.0, type=float,
                    help='loss weights for joint score')

# iteration number for G/D training
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--iter_G", default=1, type=int)
parser.add_argument("--iter_D", default=2, type=int)

parser.add_argument("--save_dir", default='runs_emotion', type=str)
parser.add_argument("--vgg_pretrain", action='store_true')  # raf: False / affectnet: true
parser.add_argument("--pretrain_vggface_dir", default='vgg_face_dag.pth', type=str)
# affectnet: ../datasets/affectnet/
parser.add_argument("--root", default='../datasets/rafd/basic',
                    type=str, help='path to dataset folder')
args = parser.parse_args()
###################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(writer, logger):
    if args.base_dataset == 'raf':
        img_root = os.path.join(args.root, 'Image/myaligned/imgs')
        # noisy training lbl: predicted lbl on RAF by pretrained affectnet model, label in raf style:
        # surprise, 2: fear, 3: disgust, 4: happiness, 5: sadness, 6: anger, 7: neutral
        train_exp_csv_file = 'noisy_labels/list_label_AffectnetModel.txt'
        train_va_csv_file = 'noisy_labels/list_va_AffectnetModel.txt'
        # val lbl: original lbl of RAF
        val_csv_file = os.path.join(args.root, 'EmoLabel/list_patition_label.txt')

        train_dataset = DataloaderRAF_MultiTask(img_size=args.img_size, is_transform=True, split='train')
        train_dataset.load_data_exp(train_exp_csv_file, img_root)
        train_dataset.load_data_va(train_va_csv_file)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=8)

        val_dataset = DataloaderRAF(img_size=args.img_size, is_transform=False, split='test')
        val_dataset.load_data(val_csv_file, img_root)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=4)

    elif args.base_dataset == 'affectnet':
        img_root = os.path.join(args.root, 'myaligned')
        # noisy exp training lbl: predicted by RAF pretrained model, labeled in affectnet style
        # 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger
        train_exp_csv_file = 'noisy_labels/training_RAFModel.csv'
        train_va_csv_file = os.path.join(args.root, 'training.csv')
        # test: load affectnet original validation exp lbl
        test_csv_file = os.path.join(args.root, 'validate.csv')

        train_dataset = DataloaderAffectnet_MultiTask(img_size=args.img_size, exp_classes=7, is_transform=True)
        train_dataset.load_data(train_exp_csv_file, train_va_csv_file, img_root)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=8)
        val_dataset = DataloaderAffectnet_MultiTask(img_size=args.img_size, exp_classes=7, is_transform=False)
        val_dataset.load_data(test_csv_file, test_csv_file, img_root)
        val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=2)

    model = MultiTaskModel(args)
    if args.vgg_pretrain:
        init_state_dict = model.encoder.init_vggface_params(args.pretrain_vggface_dir)
        model.encoder.load_state_dict(init_state_dict, strict=False)
    model.setup()
    time_meter = averageMeter()

    total_steps = 0
    for epoch in range(args.num_epoch):
        for step, data in tqdm(enumerate(train_dataloader)):
            start_ts = time.time()
            total_steps += 1

            model.set_input(data)
            model.optimize_params(epoch)
            time_meter.update(time.time() - start_ts)

            if total_steps % args.log_step == 0:
                writer.add_scalar('train/loss_exp', model.loss_class.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_exp: {:.4f}  Time/Image: {:.4f}'. \
                    format(step, epoch, model.loss_class.item(), time_meter.val / args.batch_size)
                logger.info(print_str)
                print(print_str)

                writer.add_scalar('train/loss_va', model.loss_va.item(), total_steps)
                print_str = 'Step {:d}/ Epoch {:d}]  loss_va: {:.4f}  Time/Image: {:.4f}'. \
                    format(step, epoch, model.loss_va.item(), time_meter.val / args.batch_size)
                logger.info(print_str)
                print(print_str)

                if epoch > args.gan_start_epoch:
                    writer.add_scalar('train/loss_gan', model.loss_gan.item(), total_steps)
                    print_str = 'Step {:d}/ Epoch {:d}]  loss_gan: {:.4f}  Time/Image: {:.4f}'. \
                        format(step, epoch, model.loss_gan.item(), time_meter.val / args.batch_size)
                    logger.info(print_str)
                    print(print_str)

                    writer.add_scalar('train/G_total_loss', model.loss_G.item(), total_steps)
                    print_str = 'Step {:d}/ Epoch {:d}]  G_total_loss: {:.4f}  Time/Image: {:.4f}'. \
                        format(step, epoch, model.loss_G.item(), time_meter.val / args.batch_size)
                    logger.info(print_str)
                    print(print_str)
                    time_meter.reset()

                    # D loss
                    writer.add_scalar('train/D_total_loss', model.loss_D_gan.item(), total_steps)
                    print_str = 'Step {:d}/ Epoch {:d}]  loss_D_gan: {:.4f}  Time/Image: {:.4f}'. \
                        format(step, epoch, model.loss_D_gan.item(), time_meter.val / args.batch_size)
                    logger.info(print_str)
                    print(print_str)


            if total_steps % args.val_step == 0:
                model.encoder.eval()
                total, correct, correct_clean, correct_noise = 0, 0, 0, 0
                # if not args.noise:
                with torch.no_grad():
                    for step_val, data in tqdm(enumerate(val_dataloader)):
                        img = data[0].to(device)
                        clean_lbl = data[1].to(device)
                        _, _, _, z1_enc = model.encoder(img)
                        z1_exp_enc = z1_enc[:, 0:7]

                        _, predicted = z1_exp_enc.max(1)
                        total += clean_lbl.size(0)
                        correct += predicted.eq(clean_lbl).sum().item()

                    acc = 100.0 * correct / total
                    logger.info('acc:{}'.format(acc))
                    writer.add_scalar('val/acc', acc, total_steps)
                    print('acc:', acc)

                if total_steps % args.save_step == 0:
                    save_suffix = 'last'
                    model.save_networks(save_suffix, writer.file_writer.get_logdir())
                    print("[*] models saved.")
                    logger.info("[*] models saved.")


        if epoch == args.num_epoch:
            break


if __name__ == '__main__':
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    train(writer, logger)











