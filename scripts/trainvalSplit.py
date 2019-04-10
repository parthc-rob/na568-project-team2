# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:36:14 2019

@author: Kun Sun
"""


import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
import os


def main():
    '''
    Split the 36500 raw 256x256 images into the training set (25550 images) and 
    the validation set (10950 images)
    '''
    
    imgs_dir = 'E:/避免根目录/my_dataset/val_256'   # folder contains 36500 raw 256x256 images
    out_dir = 'E:/避免根目录/my_dataset/Places365'
    num_train, num_val = 25550, 10950
    
    idx = np.arange(num_train+num_val)   
    np.random.shuffle(idx)
    names = os.listdir(imgs_dir)
    
    # copy training set
    print('Construct training set.')
    for i in range(0, num_train):
        img_name = names[idx[i]]
        source = imgs_dir + '/' + img_name
        target = out_dir + '/rawtrain/' + str(i+1) + '.jpg' 
        shutil.copyfile(source, target)
        print('Copying training image %d: '%(i+1), img_name)
    
    # copy validation set    
    print('Construct validation set.')
    for i in range(num_train, num_train+num_val):
        img_name = names[idx[i]]
        source = imgs_dir + '/' + img_name
        target = out_dir + '/rawval/' + str(i-num_train+1) + '.jpg'
        shutil.copyfile(source, target)
        print('Copying validation image %d: '%(i-num_train+1), img_name)
   
    return

if __name__ == '__main__':
    main()