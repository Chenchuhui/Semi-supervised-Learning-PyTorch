import logging
import math
import numpy as np
import random
from PIL import Image
from torchvision import datasets, transforms

from .augmentation import RandAugment

logger = logging.getLogger(__name__)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2470, 0.2435, 0.2616)

def get_stl10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=96, padding=12, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])
    transform_val = transforms.Compose([
        transforms.Lambda(lambda img: Image.fromarray(np.transpose(img, (1, 2, 0)))),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])

    train_labeled_dataset = STL10SSL(root, split='train', download=True, transform=transform_labeled)
    train_unlabeled_dataset = STL10SSL(root, split='unlabeled', download=True, transform=UnlabeledTransform(mean=stl10_mean, std=stl10_std, crop_size=args.img_size, crop_ratio=args.crop_ratio))
    test_dataset = datasets.STL10(root, split='test', transform=transform_val, download=True)
    
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class UnlabeledTransform(object):
    def __init__(self, mean, std, crop_size, crop_ratio):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=96, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=96, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
            RandAugment(n=3, m=5)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class STL10SSL(datasets.STL10):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index
