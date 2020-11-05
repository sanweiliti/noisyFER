import sys

rootPath = '/Users/siwei/Desktop/noisyFER'
sys.path.append(rootPath)

import os
from tqdm import tqdm
import cv2
import csv
import numpy as np
import argparse
from crop_align.align import MyFaceAligner


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='datasets/affectnet')
args = parser.parse_args()


def lms_to_np(lms):
    lms = lms.split(';')
    x_cor_list, y_cor_list = [], []
    for i in range(len(lms)):
        if i % 2 == 0:
            x_cor_list.append(float(lms[i]))
        else:
            y_cor_list.append(float(lms[i]))
    lms = [x_cor_list, y_cor_list]
    lms = np.asarray(lms)  # [2, 68]
    return lms


# training.csv
def crop_align_affectnet(csv_file, root):
    img_root = os.path.join(root, 'Manually_Annotated_Images')
    save_root = os.path.join(root, 'myaligned')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    my_fa = MyFaceAligner(desiredLeftEye=(0.3, 0.3), desiredFaceWidth=256)

    cnt = 0
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader):
            cur_sample = {}
            cur_sample['img_path'] = os.path.join(img_root, row['subDirectory_filePath'].split('/')[1])
            lms = row['facial_landmarks']
            cur_sample['lms'] = lms_to_np(lms)
            cur_sample['expression'] = int(row['expression'][0:])
            # for Uncertain and No-face categories the value is -2)
            cur_sample['valence'] = float(row['valence'])
            cur_sample['arousal'] = float(row['arousal'])

            # affectnet emotion label:
            # 0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt
            # 8: None, 9: Uncertain, 10: No-Face
            if cur_sample['valence'] != -2 and 0 <= cur_sample['expression'] <= 7:
                img = cv2.imread(cur_sample['img_path'])
                img_name = row['subDirectory_filePath'].split('/')[1]
                save_path = os.path.join(save_root, img_name)
                # use 68 lms provided by AffectNet
                result = my_fa.align(img, cur_sample['lms'])
                cv2.imwrite(save_path, result)
                cnt += 1

    print('num of saved images:', cnt)


if __name__ == '__main__':
    print('crop and align for affectnet training set...')
    crop_align_affectnet(csv_file=os.path.join(args.root, 'training.csv'), root=args.root)
    print('crop and align for affectnet validation set...')
    crop_align_affectnet(csv_file=os.path.join(args.root, 'validate.csv'), root=args.root)