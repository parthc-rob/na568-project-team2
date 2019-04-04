# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 01:39:40 2019

@author: Kun Sun
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def randViewpointWarp(im, w, h, s=4.0):
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



if __name__ == "__main__":
    #img1 = mpimg.imread('C:/Users/hp/Pictures/Saved Pictures/test1.jpg')
    img1 = mpimg.imread('test_images/test1.jpg')
    img2 = randViewpointWarp(img1,img1.shape[1],img1.shape[0])
    ImagePairShow(img1, img2)