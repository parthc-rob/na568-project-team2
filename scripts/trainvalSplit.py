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
    Split raw RGB images into the training set (70% of total RGB images) and 
    the validation set (30% of total RGB images)
    '''
    
    # your path for a source folder containing raw RGB images
    imgs_dir = ...   
    # our path to storage folders rawtrain and rawval
    # the output path will be "out_dir + /rawtrain" and "out_dir + /rawval"
    out_dir = ...
    # number of images in folder "rawtrain" (0.7 * num_total) and "rawval" (0.3 * num_total)
    num_train, num_val = ..., ...
    
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
