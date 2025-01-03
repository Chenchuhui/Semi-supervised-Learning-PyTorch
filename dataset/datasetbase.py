# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import numpy as np 
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset

from .augmentation import RandAugment
from .utils import get_onehot


class PreloadBasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 alg,
                 w_data1,
                 w_data2=None,
                 m_data=None,
                 s_data1=None,
                 s_data2=None,
                 targets=None,
                 transform=None,
                 num_classes=None,
                 is_ulb=False,
                 onehot=False,
                 show_img=False,
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
        super(PreloadBasicDataset, self).__init__()
        self.alg = alg
        self.w_data1 = w_data1
        self.w_data2 = w_data2
        self.m_data = m_data
        self.s_data1 = s_data1
        self.s_data2 = s_data2
        self.targets = targets
        self.transform = transform

        self.num_classes = num_classes
        self.is_ulb = is_ulb
        self.onehot = onehot
        self.show_img = show_img
    
    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target_ = self.targets[idx]
            target = target_ if not self.onehot else get_onehot(self.num_classes, target_)

        # set augmented images
        img_w1 = self.w_data1[idx] if self.w_data1 is not None else None
        img_w2 = self.w_data2[idx] if self.w_data2 is not None else None
        img_m = self.m_data[idx] if self.m_data is not None else None
        img_s1 = self.s_data1[idx] if self.s_data1 is not None else None
        img_s2 = self.s_data2[idx] if self.s_data2 is not None else None
        return img_w1, img_w2, img_m, img_s1, img_s2, target

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img_w1, img_w2, img_m, img_s1, img_s2, target = self.__sample__(idx)
        if not self.is_ulb:
            img_w1 = self.transform(img_w1) if img_w1 is not None else None
            # if self.show_img:
            #     import matplotlib.pyplot as plt
            #     import numpy as np
            #     img_np = img_w1.numpy().transpose(1, 2, 0)  # Convert from CHW to HWC format
            #     # If normalized, denormalize for proper visualization
            #     img_np = img_np * 255 if img_np.max() <= 1 else img_np  # Rescale pixel values if needed
            #     img_np = img_np.astype(np.uint8)

            #     # Display the image and target
            #     plt.imshow(img_np)
            #     plt.title(f"Target: {target}")
            #     plt.axis('off')  # Hide axes for better visualization
            #     plt.show()
            return {'x_lb': img_w1, 'y_lb': target} 
        else:
            if self.alg == 'fullysupervised' or self.alg == 'supervised':
                return {'idx_ulb': idx}
            elif self.alg == 'pseudolabel' or self.alg == 'vat':
                return {'idx_ulb': idx, 'x_ulb_w':img_w} 
            elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                # NOTE x_ulb_s here is weak augmentation
                img_w1 = self.transform(img_w1) if img_w1 is not None else None
                img_w2 = self.transform(img_w2) if img_w2 is not None else None
                return {'idx_ulb': idx, 'x_ulb_w': img_w1, 'x_ulb_s': img_w2}
            # elif self.alg == 'sequencematch' or self.alg == 'somematch':
            elif self.alg == 'sequencematch':
                return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_m': self.medium_transform(img), 'x_ulb_s': self.strong_transform(img)} 
            elif self.alg == 'remixmatch':
                img_w1 = self.transform(img_w1) if img_w1 is not None else None
                img_w2 = self.transform(img_w2) if img_w2 is not None else None
                img_s1 = self.transform(img_s1) if img_s1 is not None else None
                img_s2 = self.transform(img_s2) if img_s2 is not None else None
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
        return len(self.w_data1)

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
            # if isinstance(img, np.ndarray):
            #     img = Image.fromarray(img)
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