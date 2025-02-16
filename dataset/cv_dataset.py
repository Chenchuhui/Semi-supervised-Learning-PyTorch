from .cifar import get_cifar10, get_cifar100
from .svhn import get_svhn
from .stl10 import get_stl10

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'svhn': get_svhn,
                   'stl10': get_stl10}