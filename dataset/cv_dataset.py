from .cifar import get_cifar10, get_cifar100
from .svhn import get_svhn

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'svhn': get_svhn}