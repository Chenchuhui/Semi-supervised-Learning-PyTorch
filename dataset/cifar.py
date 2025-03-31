import logging
import math

import numpy as np
import random
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .augmentation import RandAugment

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(args.img_size * (1 - args.crop_ratio)),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = x_u_split(
        args, base_dataset.targets, args.val_split)
    
    if not args.preaug:
        train_labeled_dataset = CIFAR10SSL(
            root, train_labeled_idxs, train=True,
            transform=transform_labeled)

        train_unlabeled_dataset = CIFAR10SSL(
            root, train_unlabeled_idxs, train=True,
            transform=UnlabeledTransform(mean=cifar10_mean, std=cifar10_std, crop_size=args.img_size, crop_ratio=args.crop_ratio))
    else:
        print("----Pre-augment labeled data----")
        train_labeled_dataset = CIFAR10SSLPreaug(
            root, train_labeled_idxs, is_ulb=False, batch_size=args.batch_size, iteration=args.train_iteration, rep=args.rep, train=True, 
            transform=transform_labeled)

        print("----Pre-augment unlabeled data----")
        train_unlabeled_dataset = CIFAR10SSLPreaug(
            root, train_unlabeled_idxs, is_ulb=True, batch_size=args.batch_size, iteration=args.train_iteration, rep=args.rep, train=True,
            transform=UnlabeledTransform(mean=cifar10_mean, std=cifar10_std, crop_size=args.img_size, crop_ratio=args.crop_ratio))

    val_dataset = CIFAR10SSL(
            root, val_idxs, train=True,
            transform=transform_val)
    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs, val_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=UnlabeledTransform(mean=cifar100_mean, std=cifar100_std, crop_size=args.img_size, crop_ratio=args.crop_ratio))
    
    val_dataset = CIFAR100SSL(
        root, val_unlabeled_idxs, train=True,
        transform=transform_val)
    
    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

def x_u_split(args, labels, split=0.1):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    val_per_class = int(len(labels)*split) // args.num_classes
    labeled_idx = []
    val_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        l_idx = np.random.choice(idx, label_per_class, False)
        remaining_idx = np.setdiff1d(idx, l_idx)
        v_idx = np.random.choice(remaining_idx, val_per_class, False)
        labeled_idx.extend(l_idx)
        val_idx.extend(v_idx)
    labeled_idx = np.array(labeled_idx)
    val_idx = np.array(val_idx)
    assert len(labeled_idx) == args.num_labeled
    assert len(val_idx) == len(labels)*split

    unlabeled_idx = np.setdiff1d(unlabeled_idx, np.hstack([val_idx]))

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.train_iteration / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
        # num_expand_ulb_x = math.ceil(
        #     args.batch_size * args.train_iteration / len(labels))
        # unlabeled_idx = np.hstack([unlabeled_idx for _ in range(num_expand_ulb_x)])
    np.random.shuffle(labeled_idx)
    np.random.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx, val_idx


class UnlabeledTransform(object):
    def __init__(self, mean, std, crop_size, crop_ratio):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(crop_size * (1 - crop_ratio)),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(crop_size * (1 - crop_ratio)),
                                  padding_mode='reflect'),
            RandAugment(n=3, m=5)
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class CIFAR10SSLPreaug(datasets.CIFAR10):
    def __init__(self, root, indexs, is_ulb, batch_size, iteration, 
                 rep, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        self.is_ulb = is_ulb
        self.rep = rep
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.aug_factor = math.ceil(batch_size * iteration / len(self.data))
        
        # Pre-aug
        self.w_data, self.s_data = self.transform_data()

    def transform_data(self):
        w_data_lst = [[] for _ in range(len(self.data))]
        s_data_lst = [[] for _ in range(len(self.data))] if self.is_ulb else None

        for i, img in enumerate(self.data):
            img = Image.fromarray(img)
            try:
                if not self.is_ulb:
                    for _ in range(self.aug_factor * self.rep):
                        w_data_lst[i].append(self.transform(img))
                else:
                    for _ in range(self.aug_factor * self.rep):
                        inputs_u_w, inputs_u_s = self.transform(img)
                        w_data_lst[i].append(inputs_u_w)
                        s_data_lst[i].append(inputs_u_s)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")

        return w_data_lst, s_data_lst


    def __getitem__(self, index):
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.is_ulb:
            img = random.choice(self.w_data[index])
            return img, target
        else:
            w_img = random.choice(self.w_data[index])
            s_img = random.choice(self.s_data[index])
            return (w_img, s_img), target

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class CIFAR100SSLPreaug(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
