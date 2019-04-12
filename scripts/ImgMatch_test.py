# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 17:32:01 2019

Test on the CampusLoopDataset
"""


import numpy as np
import cv2
import os
import matplotlib.image as mpimg
from keras import models
from AcrossChannelNorm import LRN


def resizeImg(img, rw=160, rh=120):
    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_CUBIC)   # resize the image
    return img


def predict_on_keyframe(keyframe, model):
    '''
    For a given keyframe (RGB image), use the pretrained model to
    predict the HOG feature of the keyframe.
    Params:
        keyframe: a RGB image in numpy array format
        model: pretrained keras model object
    Output:
        hog: a 1D numpy array 1*3648
    '''
    
   
    # conver keyframe into gray scale
    gray = cv2.cvtColor(keyframe, cv2.COLOR_RGB2GRAY)
    # resize to fot into network input size 120*160
    gray = resizeImg(gray)
   
    # expand gray format from h*w to h*w*1
    x = np.expand_dims(gray, axis=-1)
    
    list = []
    list.append(x)
    # expand from h*w*1 to 1*h*w*1
    x = np.stack(list, axis=0)
    
    # predict the hog feature
    hog = model.predict(x)   
    
    return hog


def predict_on_CampusLoopDataset():
    imgs1_path = 'E:/避免根目录/my_dataset/CampusLoopDataset/live'
    out1_path = 'E:/避免根目录/my_dataset/CampusLoopDataset/livehogs'
    imgs2_path = 'E:/避免根目录/my_dataset/CampusLoopDataset/memory'
    out2_path = 'E:/避免根目录/my_dataset/CampusLoopDataset/memoryhogs'
    model_path = 'E:/JuWorkDir/568_project/calc_model.h5'
    
    # load the pretrained network model
    print('Loading the trained model from: ', model_path)
    final_model = models.load_model(model_path,custom_objects={'LRN': LRN})
    
    # predict on folder CampusLoopDataset/live
    names = os.listdir(imgs1_path)
    num = len(names)
    for i in range(0, num):
        keyframe = mpimg.imread(imgs1_path+'/'+names[i])
        hog = predict_on_keyframe(keyframe, final_model)
        np.save(out1_path+'/'+str(i+1)+'.npy',hog)
        print('Save prediction results for %s'%('live/'+names[i]))
        
    # predict on folder CampusLoopDataset/live
    names = os.listdir(imgs2_path)
    num = len(names)
    for i in range(0, num):
        keyframe = mpimg.imread(imgs2_path+'/'+names[i])
        hog = predict_on_keyframe(keyframe, final_model)
        np.save(out2_path+'/'+str(i+1)+'.npy',hog)
        print('Save prediction results for %s'%('memory/'+names[i]))
    
    return


def load_hogSet(path):
    names = os.listdir(path)
    num = len(names)
    
    hogs = []
    for i in range(0, num):
        hog = np.load(path+'/'+names[i])
        hogs.append(hog)
    
    hogset = np.concatenate(hogs,axis=0)
    
    return hogset
    

def l2_distance(x1, x2):
    '''
    compute the euclidean distance between to 1D numpy array
    '''
    dist = np.sum(np.square(x1-x2), axis=0)
    dist = np.sqrt(dist)
    return dist


def nn_search(x, Y):
    '''
    Search the nearest row in Y to x accroding to l2 distance 
    Params:
        x: 1D numpy array
        Y: 2D numpy array
    '''
    n = Y.shape[0]   # number of row vectors in Y
    map = np.zeros((n,2))   # distances and labels
    for i in range(0, n):
        #map[i][0] = l2_distance(x, Y[i,:])
        map[i][0] = np.linalg.norm(x - Y[i,:])
        map[i][1] = i+1
    # sorted map according to the 1st column (distances)
    map = map[map[:,0].argsort()]
    
    # return the label of the nearest element
    return map[0][1]    


def match(set1, set2):
    n = set1.shape[0]
    # label in set1 and the matched label in set2
    results = np.zeros((n,2))
    for i in range(0, n):
        results[i][0] = i+1
        results[i][1] = nn_search(set1[i,:],set2)
    return results

if __name__ == '__main__':
    
    predict_on_CampusLoopDataset()
    
    set1 = load_hogSet('E:/避免根目录/my_dataset/CampusLoopDataset/livehogs')
    set2 = load_hogSet('E:/避免根目录/my_dataset/CampusLoopDataset/memoryhogs')
    
    matching = match(set1,set2)
    print(matching)
    print(matching[:,0]-matching[:,1])