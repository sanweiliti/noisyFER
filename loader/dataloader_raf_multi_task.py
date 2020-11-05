import torch
import scipy.misc as m
import os
import csv
import numpy as np
from tqdm import tqdm
from torch.utils import data
import cv2
from transforms import initAlignTransfer
import time

# RAF: 1: surprise, 2: fear, 3: disgust, 4: happiness, 5: sadness, 6: anger, 7: neutral
# Affectnet: 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
transition_to_affectnet = dict()
transition_to_affectnet[1] = 3
transition_to_affectnet[2] = 4
transition_to_affectnet[3] = 5
transition_to_affectnet[4] = 1
transition_to_affectnet[5] = 2
transition_to_affectnet[6] = 6
transition_to_affectnet[7] = 0

class DataloaderRAF_MultiTask(data.Dataset):
    def __init__(self, img_size=256, is_transform=False, split='train'):
        self.img_size = img_size
        self.split = split
        self.is_transform = is_transform
        self.transform = initAlignTransfer(self.img_size, crop_size=self.img_size)
        self.img_list = []
        self.exp_lbl_list = []
        self.val_lbl_list, self.aro_lbl_list = [], []

    # label file: list_patition_label.txt
    # img file: rafd/basic/Image/aligned/imgs
    def load_data_exp(self, csv_file, img_root):
        num = 0
        with open(csv_file, 'r') as csvfile:
            reader = csvfile.readlines()
            if self.split == 'train':
                reader = reader[0:12271]
            else:
                reader = reader[12271:]
            for row in tqdm(reader):
                num += 1
                img_name = row[0:-3]
                img_dir = os.path.join(img_root, img_name[0:-4] + "_aligned" + img_name[-4:])
                exp_original_lbl = int(row[-2:-1])
                exp_trans_lbl = transition_to_affectnet[exp_original_lbl]
                self.img_list.append(img_dir)
                self.exp_lbl_list.append(exp_trans_lbl)

    def load_data_va(self, csv_file):
        with open(csv_file, 'r') as csvfile:
            reader = csvfile.readlines()
            if self.split == 'train':
                reader = reader[0:12271]
            else:
                reader = reader[12271:]
            for row in tqdm(reader):
                temp = row.split(' ')
                valence, arousal = float(temp[1]), float(temp[2][:-1])
                if valence >= 1:
                    valence = 1.0
                if valence <= -1:
                    valence = -1.0
                if arousal >= 1:
                    arousal = 1.0
                if arousal <= -1:
                    arousal = -1.0
                self.val_lbl_list.append(valence)
                self.aro_lbl_list.append(arousal)
        # if len(self.img_list) != len(self.val_lbl_list):
        #     print('!!!!!!!')
        print('preprocessing completed, find {} useful images'.format(len(self.img_list)))


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = cv2.imread(img_path, 1)  # BGR
        # img = img[:, :, ::-1]  # BGR --> RGB
        img = cv2.resize(img, (self.img_size, self.img_size))
        if self.is_transform:
            img = self.transform(img)

        img = img.transpose((2, 0, 1))  # [H,W,C] --> [C,H,W]
        img = ((img / 255.0 - 0.5) / 0.5)  # normalize to [-1, 1]
        img = torch.from_numpy(img).float()

        exp_lbl = self.exp_lbl_list[index]
        va_lbl = np.array([self.val_lbl_list[index], self.aro_lbl_list[index]])
        va_lbl = torch.from_numpy(va_lbl).float()

        return img, exp_lbl, va_lbl, img_path

