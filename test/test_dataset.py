#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.datasets as dset
import torchvision.transforms as transforms
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset import *

def test_create_dataset():
    ds = create_dataset('mnist')
    print('mnist len:', len(ds))

    __import__('IPython').embed()

    ds = create_dataset('cifar10', image_size=224)
    print('cifar10 len:', len(ds))
    ds = create_dataset('cifar100', image_size=224)
    print('cifar100 len:', len(ds))


if __name__ == "__main__":
    test_create_dataset()
