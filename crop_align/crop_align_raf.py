import sys

rootPath = '/Users/siwei/Desktop/noisyFER'
sys.path.append(rootPath)

import os
from tqdm import tqdm
import cv2
import numpy as np
import argparse
from crop_align.align import MyFaceAligner_RAF



parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='datasets/rafd/basic')
args = parser.parse_args()


def crop_align_raf(root):
    img_root = os.path.join(root, 'Image/original/imgs')
    csv_file = os.path.join(root, 'EmoLabel/list_patition_label.txt')
    lms_root = os.path.join(root, 'Annotation/manual')
    save_root = os.path.join(root, 'Image/myaligned/imgs')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    my_fa = MyFaceAligner_RAF(desiredLeftEye=(0.3, 0.3), desiredFaceWidth=256)

    cnt = 0
    with open(csv_file, 'r') as csvfile:
        reader = csvfile.readlines()
        for row in tqdm(reader):
            img_name = row[0:-3]
            img_dir = os.path.join(img_root, img_name)
            img = cv2.imread(img_dir)
            lms_file_path = os.path.join(lms_root, img_name[0:-4] + "_manu_attri.txt")
            with open(lms_file_path, 'r') as lmsfile:
                line = lmsfile.readlines()
                left_eye_x = float(line[0][0:7])
                left_eye_y = float(line[0][-8:-1])
                right_eye_x = float(line[1][0:7])
                right_eye_y = float(line[1][-8:-1])
                lms = np.array([[left_eye_x, right_eye_x], [left_eye_y, right_eye_y]])

            result = my_fa.align(img, lms)
            align_img_name = img_name[0:-4] + "_aligned" + img_name[-4:]
            save_path = os.path.join(save_root, align_img_name)
            cv2.imwrite(save_path, result)
            cnt += 1

    print('num of saved images:', cnt)


if __name__ == '__main__':
    print('crop and align for raf...')
    crop_align_raf(root=args.root)