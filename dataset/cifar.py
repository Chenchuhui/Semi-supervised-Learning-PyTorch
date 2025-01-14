# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torch
import torchvision
import numpy as np
import math
import cv2

from torchvision import transforms
import torchvision.transforms.functional as F
from .datasetbase import BasicDataset, PreAugBasicDataset
from .augmentation import RandAugment, RandomResizedCropAndInterpolation
from .utils import split_ssl_data


mean, std = {}, {}
mean['cifar10'] = [0.4914, 0.4822, 0.4465]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.2471, 0.2435, 0.2616]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x).float()
        return x
    
def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def preprocess_cifar_transforms(args):
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
    ])

    transform_original = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform_weak, transform_medium, transform_strong, transform_original

def get_cifar_transforms(args):
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[args.dataset], std[args.dataset])
    ])

    return transform_weak, transform_medium, transform_strong, transform_val

def load_local_data(folder):
        images, labels = [], []
        for file_name in os.listdir(folder):
            if file_name.endswith(".png"):
                label = int(file_name.split("_")[0])  # Assumes "label_index.png" naming format
                img = cv2.imread(os.path.join(folder, file_name))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                images.append(img)
                labels.append(label)
        images = transpose(np.stack(images))
        labels = np.array(labels)
        return images, labels

def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', preaug=False):
    data_dir = os.path.join(data_dir, name.lower())
    w_t, m_t, s_t, v_t = get_cifar_transforms(args)
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets
    data, targets = np.array(data), np.array(targets)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(targets, int(num_labels/10))
    lb_data, lb_targets, ulb_data, ulb_targets = data[train_labeled_idxs], targets[train_labeled_idxs], data[train_unlabeled_idxs], targets[train_unlabeled_idxs]
    val_data, val_targets = data[val_idxs], targets[val_idxs]
    
    def count_lb_ulb(lb_targets, ulb_targets):
        lb_count = [0 for _ in range(num_classes)]
        ulb_count = [0 for _ in range(num_classes)]
        for c in lb_targets:
            lb_count[c] += 1
        for c in ulb_targets:
            ulb_count[c] += 1
        print("lb count: {}".format(lb_count))
        print("ulb count: {}".format(ulb_count))
    count_lb_ulb(lb_targets, ulb_targets)
    # if alg == 'fullysupervised':
    #     lb_data = data
    #     lb_targets = targets
    if preaug:
        lb_dset = PreAugBasicDataset(alg, lb_data, lb_targets, num_classes, w_t, m_t, s_t, False, False, batch_size=args.batch_size, iteration=args.train_iteration, rep=args.lb_rep)
        ulb_dset = PreAugBasicDataset(alg, ulb_data, ulb_targets, num_classes, w_t, m_t, s_t, True, False, batch_size=args.batch_size, iteration=args.train_iteration, rep=args.ulb_rep)
        val_dset = PreAugBasicDataset(alg, val_data, val_targets, num_classes, v_t, None, None, False, False, batch_size=1, iteration=1, rep=1)

    else:
        lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, w_t, False, m_t, s_t, False)
        ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, w_t, True, m_t, s_t, False)
        val_dset = BasicDataset(alg, val_data, val_targets, num_classes, v_t, False, None, None, False)

    # Load test dataset
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets
    test_data, test_targets = np.array(test_data), np.array(test_targets)
    test_dset = BasicDataset(alg, test_data, test_targets, num_classes, v_t, False, None, None, False)

    return lb_dset, ulb_dset, val_dset, test_dset

# def save_images(dataset, transform_fn, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
    
#     for i, (img, label) in enumerate(dataset):
#         transformed_img = transform_fn(img) if transform_fn is not None else img
#         save_path = os.path.join(output_dir, f"{label}_{i}.png")
#         cv2.imwrite(save_path, cv2.cvtColor(transformed_img.numpy().transpose(1, 2, 0) * 255, cv2.COLOR_RGB2BGR))

# # Execute this file if needs data preprocessing
# def main(name, data_dir='./data'):
#     # Define output directories
#     original_dir = "./data/augmented/original"
#     weak_dir1 = "./data/augmented/weak1"
#     weak_dir2 = "./data/augmented/weak2"
#     medium_dir = "./data/augmented/medium"
#     strong_dir1 = "./data/augmented/strong1"
#     strong_dir2 = "./data/augmented/strong2"
#     test_dir = "./data/augmented/test"

#     data_dir = os.path.join(data_dir, name.lower())
#     dataset = getattr(torchvision.datasets, name.upper())
#     dataset_train = dataset(data_dir, train=True, download=True)
#     dataset_test = dataset(data_dir, train=False, download=True)
#     class Args:
#         dataset = name
#         img_size = 32
#         crop_ratio = 0.875
#     args = Args()
#     weak_transform, medium_transform, strong_transform, ori_transform = preprocess_cifar_transforms(args)
    
#     # Perform transformations and save images
#     print("Saving original...")
#     # save_images(dataset_train, ori_transform, original_dir)

#     print("Saving weak transformations...")
#     save_images(dataset_train, weak_transform, weak_dir1)
#     save_images(dataset_train, weak_transform, weak_dir2)

#     print("Saving medium transformations...")
#     save_images(dataset_train, medium_transform, medium_dir)

#     print("Saving strong transformations...")
#     save_images(dataset_train, strong_transform, strong_dir1)
#     save_images(dataset_train, strong_transform, strong_dir2)

#     print("Saving test...")
#     save_images(dataset_test, ori_transform, test_dir)
