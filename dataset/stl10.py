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

from torchvision import transforms, datasets
from torch.utils.data import Subset
import numpy as np
from PIL import Image

def get_stl10(args, root):
    transform_labeled = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=args.img_size, padding=int(args.img_size * (1 - args.crop_ratio)), padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])

    # Load full labeled STL10 training data
    full_labeled_dataset = datasets.STL10(root=root, split='train', download=True)
    targets = full_labeled_dataset.labels  # length = 5000

    # Generate index splits (custom function, like x_u_split for STL10)
    train_labeled_idxs, _, val_idxs = x_u_split(args, targets, args.val_split)

    # Apply transforms and subset
    train_labeled_dataset = Subset(STL10SSL(root, split='train', download=True, transform=transform_labeled), train_labeled_idxs)
    val_dataset = Subset(STL10SSL(root, split='train', download=True, transform=transform_val), val_idxs)

    # Unlabeled split from 'unlabeled' partition
    train_unlabeled_dataset = STL10SSL(
        root, split='unlabeled', download=True,
        transform=UnlabeledTransform(mean=stl10_mean, std=stl10_std, crop_size=args.img_size, crop_ratio=args.crop_ratio)
    )

    # Test set
    test_dataset = STL10SSL(root, split='test', transform=transform_val, download=True)

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
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
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
        # Ensure test split uses only transform_val
        if split == 'test' and isinstance(transform, UnlabeledTransform):
            raise ValueError("UnlabeledTransform should not be used for the test split.")
    
    def __getitem__(self, index):
        # Debug: Print index and split
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.split == 'test':
            return img, target
        
        return img, target, index
