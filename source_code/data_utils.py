"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

import random

import _pickle as pickle

########### TEMP #################
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
########### TEMP #################

# pylint: disable=C0326
SEG_LABELS_LIST = [
    {"id": 0,  "name": "building",   "rgb_values": [128, 0,    0]},
    {"id": 1,  "name": "grass",      "rgb_values": [0,   128,  0]},
    {"id": 2,  "name": "tree",       "rgb_values": [128, 128,  0]},
    {"id": 3,  "name": "cow",        "rgb_values": [0,   0,    128]},
    {"id": 4,  "name": "horse",      "rgb_values": [128, 0,    128]},
    {"id": 5,  "name": "sheep",      "rgb_values": [0,   128,  128]},
    {"id": 6,  "name": "sky",        "rgb_values": [128, 128,  128]},
    {"id": 7,  "name": "mountain",   "rgb_values": [64,  0,    0]},
    {"id": 8,  "name": "airplane",   "rgb_values": [192, 0,    0]},
    {"id": 9,  "name": "water",      "rgb_values": [64,  128,  0]},
    {"id": 10, "name": "face",       "rgb_values": [192, 128,  0]},
    {"id": 11, "name": "car",        "rgb_values": [64,  0,    128]},
    {"id": 12, "name": "bicycle",    "rgb_values": [192, 0,    128]},
    {"id": 13, "name": "flower",     "rgb_values": [64,  128,  128]},
    {"id": 14, "name": "sign",       "rgb_values": [192, 128,  128]},
    {"id": 15, "name": "bird",       "rgb_values": [0,   64,   0]},
    {"id": 16, "name": "book",       "rgb_values": [128, 64,   0]},
    {"id": 17, "name": "chair",      "rgb_values": [0,   192,  0]},
    {"id": 18, "name": "road",       "rgb_values": [128, 64,   128]},
    {"id": 19, "name": "cat",        "rgb_values": [0,   192,  128]},
    {"id": 20, "name": "dog",        "rgb_values": [128, 192,  128]},
    {"id": 21, "name": "body",       "rgb_values": [64,  64,   0]},
    {"id": 22, "name": "boat",       "rgb_values": [192, 64,   0]},
    {"id": 23, "name": "void",       "rgb_values": [0,   0,    0]}]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file, train=True, rotation='constrained', crop_size=240, crop_style='center',
                 mu=[0.6353146 , 0.6300146 , 0.52398586], std=[0.3769369 , 0.36186826, 0.36188436]):
        self.root_dir_name = os.path.dirname(image_paths_file)
        self.train = train
        self.rotation = rotation
        self.mu = mu
        self.std = std
        self.crop_size = crop_size
        self.crop_style = crop_style

        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)

    def get_item_from_index(self, index):
        normalize = transforms.Normalize(mean=self.mu,
                                 std=self.std) # my shit

        to_tensor = transforms.ToTensor()
        vflipseed = random.randint(0,2**32)
        hflipseed = random.randint(0,2**32)
        
        hflip=transforms.RandomHorizontalFlip()
        vflip=transforms.RandomVerticalFlip()
        rotseed = random.randint(0,2**32)
        rot=transforms.RandomRotation(90, expand=True)
        
        if self.crop_style == 'center':
            cropseed = random.randint(0,2**32)
            crop = transforms.CenterCrop(self.crop_size) # my commenting shit ##############################################
        elif self.crop_style == 'random':
            cropseed = random.randint(0,2**32)
            crop = transforms.RandomCrop(self.crop_size, pad_if_needed=True) # THE RANDOM CROP
        # rand_jitter=transforms.ColorJitter()
        # rand_hflip=transforms.RandomHorizontalFlip()
        # rand_vflip=transforms.RandomVerticalFlip()
        # rand_rot=transforms.RandomRotation()
        
        # x=int (np.random.uniform(0,360,1))
        # image=image.rorate(image,x)
        # label=label.rorate(label,x)
        
        img_id = self.image_names[index].replace('.bmp', '')

        img = Image.open(os.path.join(self.root_dir_name,
                                      'images',
                                      img_id + '.bmp')).convert('RGB')
        target = Image.open(os.path.join(self.root_dir_name,
                                         'targets',
                                         img_id + '_GT.bmp'))
        
        # Random ROTATION
        if self.train and (self.rotation=='random'):
            random.seed(rotseed)
            img = rot(img)
            random.seed(rotseed)
            target = rot(target)
        
        # Constrained rotation
        if self.train and (self.rotation=='constrained'):
            if random.random() > 0.5:
                angle = 90
                img = F.rotate(img, angle=angle, expand=True)
                target = F.rotate(target, angle=angle, expand=True)
        
        # CROP (random or normal)
        if not self.crop_style == None:
            random.seed(cropseed)
            img = crop(img)
            random.seed(cropseed)
            target = crop(target)
        # img = center_crop(img) ################################################################################
        # img=rand_jitter(img)
        
        # HORIZONTAL FLIP
        if self.train:
            random.seed(hflipseed)
            img=hflip(img)
            random.seed(hflipseed)
            target=hflip(target)
        
        # VERTICAL FLIP
        if self.train:
            random.seed(vflipseed)
            img=vflip(img)
            random.seed(vflipseed)
            target=vflip(target)
        
        img = to_tensor(img)
        img = normalize(img)
        # y = img
        #print('tensor')
        #plt.imshow(y.numpy())
        
        #print('normalized')
        #plt.imshow(img.numpy())

        # target = center_crop(target) #######################################################################
        # target = to_tensor(target) ###
        target = np.array(target, dtype=np.int64) ###############
        
        print("original shape", target.numpy().shape)
        if target.shape[1] <= 10:
            print("original", target)
        print("original buildings", np.sum(np.all(target == np.array([[[128]], [[0]], [[0]]]), axis=0)))
        print("original sign", np.sum(np.all(target == np.array([[[192]], [[128]], [[128]]]), axis=0)))
        
        tl = target
        target_labels = target[..., 0]
        for label in SEG_LABELS_LIST:
            mask = np.all(target == label['rgb_values'], axis=2)
            target_labels[mask] = label['id']

        # # target_labels = np.transpose(target_labels)

        # print(target_labels.shape())
        
        print("labeled", target_labels.shape)
        if target.shape[1] <= 10:
            print("labeled", target_labels)
        print("labeled buildings", np.sum(target_labels == 0))
        print("labeled sign", np.sum(target_labels == 14))

        target_labels = torch.from_numpy(target_labels.copy())
        # print(target_labels.shape)
        # target_labels = target

        #target_labels = to_tensor(target_labels)

        return img, target_labels
