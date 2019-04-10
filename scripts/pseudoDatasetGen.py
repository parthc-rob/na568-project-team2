# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 01:39:40 2019
This file is used to convert a RGB images folder into a grayscale images folder (training set or validation set)
and a folder containing HOG features of each image in.

@author: Kun Sun
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import random


def resizeImg(img, rw=160, rh=120):
    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_CUBIC)   # resize the image
    return img


def randViewpointWarp(im, w=160, h=120, s=4.0):
    """
    Applies a pseudo-random perspective warp to an image.
    Params:
        im - the original image
        w, h - image width and height
        s - ratio of window size/image size 
    Returns:
        im_warp - the warped image
    """
    
    sw, sh = w/s, h/s   # window size for random points picking
    
    # four original source points
    pts_orig = np.float32([[0, 0], [0, h],
                          [w, 0], [w, h]])
    
    # four randomly picked target points
    pts_warp = np.float32([[np.random.uniform(0,sw), np.random.uniform(0,sh)], 
                           [np.random.uniform(0,sw), np.random.uniform(h-sh,h)],
                           [np.random.uniform(w-sw,w), np.random.uniform(0,sh)],
                           [np.random.uniform(w-sw,w), np.random.uniform(h-sh,h)]])
    
    # compute the afine transformation 
    M = cv2.getPerspectiveTransform(pts_warp,pts_orig)
    
    im_warp = cv2.warpPerspective(im, M, (w, h), flags=cv2.INTER_NEAREST)
    return im_warp


def ImagePairShow(img1, img2):
    """
    Display img1 and img2, used for visualizing viewpoint warping results.
    Image format: [height,width,channels]
    """
    
    f, axs = plt.subplots(1, 2, figsize=(12, 12), squeeze=False)
    f.tight_layout()
    f.subplots_adjust(hspace = 0.2, wspace = 0.1)
    axs = axs.ravel()
    
    axs[0].imshow(img1)
    axs[0].set_title('Original Image', fontsize=20)
    axs[1].imshow(img2)
    axs[1].set_title('After Viewpoint Transform', fontsize=20)
    
    return


def ImageDisplay(path, rows=2, cols=2):
    """
    Display several images in the given folder 
    """
    
    f, axs = plt.subplots(rows, cols, figsize=(12, 12), squeeze=False)
    f.tight_layout()
    f.subplots_adjust(hspace = 0.05, wspace = 0.1)
    axs = axs.ravel()
    
    n = np.int32(rows*cols)
    idx = random.sample(range(10000),n)
    for i in range(0,n):
        img = np.load(path+'/'+str(idx[i])+'.npy')
        axs[i].imshow(img[:,:,0], cmap='gray')
        axs[i].set_title(str(idx[i]), fontsize=20)
    
    return


def calcHOG(img):
    hog = cv2.HOGDescriptor((16, 32), (16,16), (16,16), (8,8), 2,1)   # configure the HOG descriptor
    
    hogVec = hog.compute(img)
    
    return hogVec


def visualizeHOG(hog1, hog2):
    rw, rh = 76, 48   # the resized hog map size
    
    hog1 = hog1.reshape((rh, rw))
    hog2 = hog2.reshape((rh, rw))
    
    ImagePairShow(hog1, hog2)
    
    return


def buildTrainSet(imgs_dir, out_dir, name):
    """
    Create a folder "train" with all training images and a folder "labels"
    with all training labels (HOG vectors).
    Params:
        imgs_dir - the path with all raw RGB images
        out_dir - the path to create folders "train"(or "val") and "trainlabels"(or "vallabels")
    """
    
    raw_imgs = os.listdir(imgs_dir)
    num = len(raw_imgs)
    
    for i in range(0, num):
        img = mpimg.imread(imgs_dir+"/"+raw_imgs[i])
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   # convert RGB to gray scale
        else:
            gray = img   # input img is already grayscale
        gray = resizeImg(gray)
        gray_warp = randViewpointWarp(gray)
                                      
        switch_flag = np.random.randint(0,2)
        if switch_flag:
            train_img = gray_warp
            train_label = calcHOG(gray)
        else:
            train_img = gray
            train_label = calcHOG(gray_warp)
          
        # change shape from 120*160 to 120*160*1
        train_img = np.expand_dims(train_img, axis=-1)
        # change shape from 3648*1 to 3648
        train_label = np.ravel(train_label)
        
        np.save(out_dir+'/'+name+'/'+str(i+1)+'.npy', train_img)
        np.save(out_dir+'/'+name+'labels/'+str(i+1)+'.npy', train_label)
        print("Save image" + str(i+1))
    
    return


if __name__ == "__main__":
    # your path for folders "rawtrain" and "rawval"
    train_imgs_dir = ".../rawtrain"
    val_imgs_dir = ".../rawval"
    # your path to storage the output folders "train" and "val"
    train_out_dir = ".../train"
    val_out_dir = ".../val"
    
    print("Construct training set...")
    buildTrainSet(train_imgs_dir, train_out_dir, 'train')
    print("Construct validation set...")
    buildTrainSet(val_imgs_dir, val_out_dir, "val")
    
    
