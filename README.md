# NoisyFER: Facial Emotion Recognition with Noisy Multi-task Annotations
Pytorch implementation for 2021 WACV paper "Facial Emotion Recognition with Noisy Multi-task Annotations".
We propose a new method for emotion prediction with noisy multi-task annotations 
by joint distribution learning
in a unified adversarial learning game.

**Facial Emotion Recognition with Noisy Multi-task Annotations** <br>
Siwei Zhang, Zhiwu Huang, Danda Pani Paudel, Luc Van Gool <br>
[[Full Paper]](https://arxiv.org/pdf/2010.09849.pdf) [[Video]](https://www.youtube.com/watch?v=bszy34vY-2o)

## Dependencies
Run `pip install -r requirements.txt` to install required dependencies.

## Datasets
Download the following datasets to `datasets/`
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [RAF-DB](http://www.whdeng.cn/RAF/model1.html)
* [AffectNet](http://mohammadmahoor.com/affectnet/)

Organize the dataset folder as:
```
Project
├── datasets
|   ├── cifar10
|   |   ├── cifar-10-batches-py
|   ├── affectnet
|   |   ├── Manually_Annotated_Images
|   |   ├── training.csv
|   |   ├── validate.csv
|   ├── rafd
|   |   ├── basic
|   |   |   ├── EmoLabel
|   |   |   |   ├── list_patition_label.txt
|   |   |   ├── Image
```

## Noisy Labels
The noisy machine labels created from pretrained models for RAF-base and AffectNet-base experiments are:
*  `noisy_labels/list_label_AffectnetModel.txt`: noisy discrete emotion labels created for RAF-base training (from pretrained model on AffectNet)
*  `noisy_labels/list_va_AffectnetModel.txt`: noisy continuous valence/arousal labels created for RAF-base training (from pretrained model on AffectNet)
*  `noisy_labels/training_RAFModel.csv`: noisy discrete emotion labels created for AffectNet-base training (from pretrained model on RAF)
* We use the original valence/arousal labels in AffectNet in AffectNet-base training.


## Crop and align images
First crop and align face images in AffectNet and RAF-DB dataset with the same crop/align protocol:
```
cd crop_align
python crop_align_affectnet.py --root datasets/affectnet
python crop_align_raf.py --root datasets/rafd/basic
```
The resulting images will be saved to `datasets/affectnet/myaligned` 
and `datasets/rafd/basic/Image/myaligned/imgs` repectively.


## Training:
### CIFAR-10
On CIFAR-10, train the model with 3 sets of different noisy labels, each set of labels with 20%/30%/40% noise ratiois:
```
python train_cifar10_inconsist_label.py --lambda_gan=0.8 --root=datasets/cifar10/cifar-10-batches-py
```

### RAF-base:
On RAF dataset, train the model with noisy emotion/va labels (which are created from AffectNet pretrained model):
```
python train_emotion_multi_task.py --img_size=64 --base_dataset=raf --lambda_gan=1.0 --gan_start_epoch=5 --root=datasets/rafd/basic
```

### AffectNet-base:
On AffectNet dataset, train the model with noisy emotion/va labels (which are created from AffectNet pretrained model):
```
python train_emotion_multi_task.py --img_size=64 --base_dataset=affectnet --lambda_gan=0.4 --gan_start_epoch=-1 --root=datasets/affectnet --vgg_pretrain
```

## Citation

When using the code/figures/data/etc., please cite our work
```
@inproceedings{zhang2020facial,
    title = {Facial Emotion Recognition with Noisy Multi-task Annotations},
    author = {Zhang, Siwei and Huang, Zhiwu and Paudel, Danda Pani and Gool, Luc Van},
    booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
    month = jan,
    year = {2021},
    month_numeric = {1}
}
```
