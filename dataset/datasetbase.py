# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import numpy as np 
from collections import deque

from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .augmentation import RandAugment
from .utils import get_onehot


class PreAugBasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 w_transform=None,
                 m_transform=None,
                 s_transform=None,
                 is_ulb=False,
                 onehot=False,
                 *args, 
                 **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(PreAugBasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets
        self.w_transform = w_transform
        self.m_transform = m_transform
        self.s_transform = s_transform

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        self.augmentation_factor = math.ceil(kwargs['batch_size'] * kwargs['iteration'] / len(self.data))

        # Data augmentation Now
        self.w_data, self.m_data, self.s_data = self.transform_data()

    def transform_data(self):
        print(self.augmentation_factor)
        w_data_lst = [deque() for _ in range(len(self.data))]
        m_data_lst = [deque() for _ in range(len(self.data))] if self.is_ulb else None
        s_data_lst = [deque() for _ in range(len(self.data))] if self.is_ulb else None

        for i, img in enumerate(self.data):
            try:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                if not self.is_ulb:
                    for _ in range(self.augmentation_factor):
                        w_data_lst[i].append(self.w_transform(img))
                else:
                    if self.alg == "mixmatch":
                        for _ in range(self.augmentation_factor * 2):
                            w_data_lst[i].append(self.w_transform(img))
                    elif self.alg == "remixmatch":
                        for _ in range(self.augmentation_factor):
                            w_data_lst[i].append(self.w_transform(img))
                            s_data_lst[i].append(self.s_transform(img))
                            s_data_lst[i].append(self.s_transform(img))
            except Exception as e:
                print(f"Error processing sample {i}: {e}")

        assert len(w_data_lst) == len(self.data), "Mismatch between data and augmented data length"
        return w_data_lst, m_data_lst, s_data_lst

    
    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        try:
            img = self.data[idx]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if self.w_data is not None and len(self.w_data[idx]) > 0:
                img_w1 = self.w_data[idx].popleft()
                self.w_data[idx].append(img_w1)  # Add it back to the end of the queue
            else:
                img_w1 = self.w_transform(img)
            
            if self.w_data is not None and len(self.w_data[idx]) > 0:
                img_w2 = self.w_data[idx].popleft()
                self.w_data[idx].append(img_w2)  # Add it back to the end of the queue
            else:
                img_w2 = None
            
            if self.m_data is not None and len(self.m_data[idx]) > 0:
                img_m = self.m_data[idx].popleft()
                self.m_data[idx].append(img_m)  # Add it back to the end of the queue
            else:
                img_m = None
            
            if self.s_data is not None and len(self.s_data[idx]) > 0:
                img_s1 = self.s_data[idx].popleft()
                self.s_data[idx].append(img_s1)  # Add it back to the end of the queue
            else:
                img_s1 = None
            
            if self.s_data is not None and len(self.s_data[idx]) > 0:
                img_s2 = self.s_data[idx].popleft()
                self.s_data[idx].append(img_s2)  # Add it back to the end of the queue
            else:
                img_s2 = None
        except IndexError:
            raise Exception(f"Not enough augmented data for index {idx}. Increase `num_augmentations` or adjust batch size/iterations.")
        return img, img_w1, img_w2, img_m, img_s1, img_s2, target
        # return img, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, img_w1, img_w2, img_m, img_s1, img_s2, target = self.__sample__(idx)
        # img, target = self.__sample__(idx)
        if not self.is_ulb:
            return {'x_lb': img_w1, 'y_lb': target} 
        else:
            if self.alg == 'fullysupervised' or self.alg == 'supervised':
                return {'idx_ulb': idx}
            elif self.alg == 'pseudolabel' or self.alg == 'vat':
                return {'idx_ulb': idx, 'x_ulb_w':img_w} 
            elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                # NOTE x_ulb_s here is weak augmentation
                return {'idx_ulb': idx, 'x_ulb_w': img_w1, 'x_ulb_s': img_w2}
            # elif self.alg == 'sequencematch' or self.alg == 'somematch':
            elif self.alg == 'sequencematch':
                return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_m': self.medium_transform(img), 'x_ulb_s': self.strong_transform(img)} 
            elif self.alg == 'remixmatch':
                if img_w1 is None or img_s1 is None or img_s2 is None:
                    raise Exception("One or more of augmented images are None")
                rotate_v_list = [0, 90, 180, 270]
                rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                return {'idx_ulb': idx, 'x_ulb_w': img_w1, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
            elif self.alg == 'comatch':
                return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)} 
            else:
                img_w1 = self.transform(img_w1) if img_w1 is not None else None
                img_s1 = self.transform(img_s1) if img_s1 is not None else None
                return {'idx_ulb': idx, 'x_ulb_w': img_w1, 'x_ulb_s': img_s1} 


    def __len__(self):
        return len(self.data)

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 data,
                 targets=None,
                 num_classes=None,
                 transform=None,
                 is_ulb=False,
                 medium_transform=None,
                 strong_transform=None,
                 onehot=False,
                 *args, 
                 **kwargs):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()
        self.alg = alg
        self.data = data
        self.targets = targets

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot

        self.transform = transform
        self.strong_transform = strong_transform
        self.medium_transform = medium_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"
    
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires medium augmentation"
    
    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img = self.data[idx]
        return img, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.__sample__(idx)

        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target} 
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w':img_w} 
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                # elif self.alg == 'sequencematch' or self.alg == 'somematch':
                elif self.alg == 'sequencematch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_m': self.medium_transform(img), 'x_ulb_s': self.strong_transform(img)} 
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)} 
                else:
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)} 


    def __len__(self):
        return len(self.data)