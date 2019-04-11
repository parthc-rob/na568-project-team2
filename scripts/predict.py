# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:27:51 2019

Use the pretrained network model to predict the HOG feature of 
a given keyframe image.
"""


import numpy as np
import cv2
import matplotlib.image as mpimg
from keras import models
from AcrossChannelNorm import LRN


def resizeImg(img, rw=160, rh=120):
    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_CUBIC)   # resize the image
    return img

def predict_on_keyframe(keyframe, model_path):
    '''
    For a given keyframe (RGB image), use the pretrained model to
    predict the HOG feature of the keyframe.
    Params:
        keyframe: a RGB image in numpy array format
    Output:
        hog: a 1D numpy array 1*3648
    '''
    
    # load the pretrained network model
    print('Loading the trained model from: ', model_path)
    final_model = models.load_model(model_path,custom_objects={'LRN': LRN})
   
    # conver keyframe into gray scale
    gray = cv2.cvtColor(keyframe, cv2.COLOR_RGB2GRAY)
    # resize to fot into network input size 120*160
    gray = resizeImg(gray)
    print(gray.shape)
    # expand gray format from h*w to h*w*1
    x = np.expand_dims(gray, axis=-1)
    print(x.shape)
    list = []
    list.append(x)
    # expand from h*w*1 to 1*h*w*1
    x = np.stack(list, axis=0)
    print(x.shape)
    
    # predict the hog feature
    hog = final_model.predict(x)   
    
    return hog

if __name__ == '__main__':
    img_path = 'E:/避免根目录/my_dataset/val_256/Places365_val_00000001.jpg'
    model_path = 'E:/JuWorkDir/568_project/calc_model.h5'
    
    keyframe = mpimg.imread(img_path)
    hog = predict_on_keyframe(keyframe, model_path)
    print(hog)