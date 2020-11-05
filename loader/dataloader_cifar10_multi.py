import pickle
import os
from torch.utils import data
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
from collections import Counter


class DataloaderCifar10_MultiLabel(data.Dataset):
    def __init__(self, img_size=32, is_transform=False, split='train', noise_ratio_list=[0.2, 0.3, 0.4]):
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.transform_train = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # HWC --> CHW, 0~255->[0.0,1.0]
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # HWC --> CHW, 0~255->[0.0,1.0]
        ])
        self.data_list = []
        self.noise_ratio_list = noise_ratio_list

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_data(self, data_root):
        all_labels = []
        all_data = []
        if self.split != 'test':
            file_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        else:
            file_list = ['test_batch']
        for i, file_name in enumerate(file_list):
            cur_batch = self.unpickle(os.path.join(data_root, file_name))
            data = cur_batch[b'data']  # [10000, 3072(32*32*3)] array
            labels = cur_batch[b'labels']  # [10000] list
            all_data.append(data)
            all_labels = all_labels + labels
        all_data = np.concatenate(all_data, axis=0)
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32)  # [num_img, 3, 32, 32], RGB
        all_data = all_data.transpose((0, 2, 3, 1))  # CHW --> HWC
        all_data = list(all_data)

        self.data_list = all_data
        self.label_list = all_labels

        self.noise_label_group = [[] for _ in range(len(self.noise_ratio_list))] # [[], [], []]
        self.major_vote_label = []  # majority vote as the new label set

        random.seed(100)
        if self.split != 'test':
            for p, noise_ratio in enumerate(self.noise_ratio_list):
                for i in range(len(self.label_list)):
                    flip = False
                    flip_rdn = random.uniform(0., 1.)
                    if flip_rdn < noise_ratio:
                        flip = True
                    if flip:
                        fake_label = random.randint(0, 9)  # [0,9]
                        self.noise_label_group[p].append(fake_label)
                    else:
                        self.noise_label_group[p].append(self.label_list[i])

            for i in range(len(self.label_list)):
                cur_list = [self.noise_label_group[0][i], self.noise_label_group[1][i], self.noise_label_group[2][i]]
                major_label = Counter(cur_list).most_common(1)[0][0]
                self.major_vote_label.append(major_label)

        # print(sum(self.label_list), sum(self.noise_label_list))
        return self.data_list, self.label_list, self.noise_label_group, self.major_vote_label


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):
        img = self.data_list[index]  # HWC, RGB, array
        img = Image.fromarray(img)
        clean_label = self.label_list[index]
        if self.split == 'train':
            noise_label_set = [self.noise_label_group[i][index] for i in range(len(self.noise_ratio_list))]
            major_label = self.major_vote_label[index]
            if self.is_transform:
                img = self.transform_train(img)  # CHW, RGB,[0,1]
        else:
            noise_label_set = []
            major_label = []
            img = self.transform_test(img)
        img = ((img - 0.5) / 0.5)  # normalize to [-1, 1]
        return img, noise_label_set, clean_label, major_label




if __name__ == '__main__':
    root = '/srv/beegfs02/scratch/emotion_perception/data/csevim/datasets/cifar10/cifar-10-batches-py'
    dataset = DataloaderCifar10_multi()
    dataset.load_data(root)





