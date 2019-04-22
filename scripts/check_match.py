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


def predict_on_keyframe(keyframe, model, preprocess):
    '''
    For a given keyframe (RGB image), use the pretrained model to
    predict the HOG feature of the keyframe.
    Params:
        keyframe: a RGB image in numpy array format
        model: pretrained keras model object
    Output:
        desciptor: a 1D numpy array 1*936
    '''
    
    if preprocess==True:
        img_yuv = cv2.cvtColor(keyframe, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        keyframe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # conver keyframe into gray scale
    gray = cv2.cvtColor(keyframe, cv2.COLOR_RGB2GRAY)
    # resize to fot into network input size 120*160
    gray = resizeImg(gray)
   
    gray = gray / 255.0
    # expand gray format from h*w to h*w*1
    x = np.expand_dims(gray, axis=-1)
    
    list = []
    list.append(x)
    # expand from h*w*1 to 1*h*w*1
    x = np.stack(list, axis=0)
    
    # predict the hog feature
    descriptor = model.predict(x) 
    
    return descriptor


def predict_on_CampusLoopDataset():
    imgs1_path = 'E:/UnderRoot/my_dataset/CampusLoopDataset/live'
    out1_path = 'E:/UnderRoot/my_dataset/CampusLoopDataset/livehogs'
    imgs2_path = 'E:/UnderRoot/my_dataset/CampusLoopDataset/memory'
    out2_path = 'E:/UnderRoot/my_dataset/CampusLoopDataset/memoryhogs'
    model_path = 'E:/JuWorkDir/568_project/calc_model_6Million.h5'
    preprocess = True
    # load the pretrained network model
    print('Loading the trained model from: ', model_path)
    my_model = models.load_model(model_path,custom_objects={'LRN': LRN})
    print('Only grap the descriptor layer.')
    deploy_model = models.Model(inputs=my_model.input,
            outputs=my_model.get_layer('deploy').output)
    
    # predict on folder CampusLoopDataset/live
    names = os.listdir(imgs1_path)
    num = len(names)
    for i in range(0, num):
        keyframe = mpimg.imread(imgs1_path+'/'+names[i])
        descriptor = predict_on_keyframe(keyframe, deploy_model, preprocess)
        descriptor = descriptor / np.linalg.norm(descriptor)
        np.save(out1_path+'/'+str(i+1)+'.npy',descriptor)
        if i%4 == 0:
            print('Save prediction results for %s'%('live/'+names[i]))
        
    # predict on folder CampusLoopDataset/live
    names = os.listdir(imgs2_path)
    num = len(names)
    for i in range(0, num):
        keyframe = mpimg.imread(imgs2_path+'/'+names[i])
        descriptor = predict_on_keyframe(keyframe, deploy_model, preprocess)
        descriptor = descriptor / np.linalg.norm(descriptor)
        np.save(out2_path+'/'+str(i+1)+'.npy',descriptor)
        if i%4 == 0:
            print('Save prediction results for %s'%('memory/'+names[i]))
    
    return


def load_DescrSet(path):
    names = os.listdir(path)
    num = len(names)
    
    descrs = []
    for i in range(0, num):
        descr= np.load(path+'/'+names[i])
        descrs.append(descr)
    
    descrset = np.concatenate(descrs,axis=0)
    
    return descrset
    

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
    return map[0:5, 0:2] 


def sim_search(x, Y):
    '''
    Search the row in Y which has the highest inner product
    with x accroding 
    Params:
        x: 1D numpy array
        Y: 2D numpy array
    '''
    n = Y.shape[0]   # number of row vectors in Y
    map = np.zeros((n,2))   # distances and labels
    for i in range(0, n):
        map[i][0] = np.dot(Y[i,:], x.T)
        map[i][1] = i+1
    # sorted map according to the 1st column (distances)
    map = map[map[:,0].argsort()]
    return map[n-5:n, 0:2]


def match(set1, set2, metric='nn'):
    n = set1.shape[0]
    # label in set1 and the matched label in set2
    results = np.zeros((n,7))
    if metric == 'nn':
        for i in range(0, n):
            results[i][0] = i+1
            neighbours = nn_search(set1[i,:],set2)
            results[i][1] = neighbours[0][1]
            results[i][2] = neighbours[1][1]
            results[i][3] = neighbours[2][1]
            results[i][4] = neighbours[3][1]
            results[i][5] = neighbours[4][1]
            results[i][6] = neighbours[0][0]   # l2 distances
    else:
        for i in range(0, n):
            results[i][0] = i+1
            neighbours = sim_search(set1[i,:],set2)
            results[i][1] = neighbours[4][1]
            results[i][2] = neighbours[3][1]
            results[i][3] = neighbours[2][1]
            results[i][4] = neighbours[1][1]
            results[i][5] = neighbours[0][1]
            results[i][6] = neighbours[4][0]   # inner product
    return results


def evaluate(matching):
    n = matching.shape[0]
    num_correct = 0
    for i in range(0, n):
        if matching[i][0] == matching[i][1]:
            print('Correct: ',matching[i,0],' ',matching[i,1:4],' metric_value: ',matching[i,6])
            num_correct += 1
    print('Total num correct: ', num_correct)
    return

if __name__ == '__main__':
    
    predict_on_CampusLoopDataset()
    
    set1 = load_DescrSet('E:/UnderRoot/my_dataset/CampusLoopDataset/livehogs')
    set2 = load_DescrSet('E:/UnderRoot/my_dataset/CampusLoopDataset/memoryhogs')
    
    matching = match(set1,set2,'nn')
    print(matching[:,0:6])
    
    evaluate(matching)