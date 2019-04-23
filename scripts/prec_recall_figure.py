# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:33:50 2019

@author: Kun Sun

To test our model on KITTI sequences
"""


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import models
from AcrossChannelNorm import LRN


def resizeImg(img, rw=160, rh=120):
    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_CUBIC)   
    return img


def predict_on_keyframe(keyframe, model):
    '''
    For a given keyframe (grayscale image), use the pretrained model to
    predict the HOG feature of the keyframe.
    Params:
        keyframe: a RGB image in numpy array format
        model: pretrained keras model object
    Output:
        descr: a 1D numpy array 1*936
    '''
    
    gray = resizeImg(keyframe, rw=160, rh=120) 
    gray = gray / 255.0   # rescale the input       
   
    # add dim "channels": expand gray format from h*w to h*w*1
    x = np.expand_dims(gray, axis=-1)
    
    # add dim "num": expand from h*w*1 to 1*h*w*1
    list = []
    list.append(x)
    x = np.stack(list, axis=0)
    
    # predict the hog feature
    descr = model.predict(x)   
    return descr


def predict_on_KITTI():
    '''
    Generate a HOG vector for each image in the NewCollege dataset
    using pretrained model
    '''
    
    global imgs_path
    global preds_path 
    global model_path 
    
    # load the pretrained network model
    print('Loading the trained model from: ', model_path)
    final_model = models.load_model(model_path,custom_objects={'LRN': LRN})
    print('Only grab the descriptor layer outputs')
    deploy_model = models.Model(inputs=final_model.input,
            outputs=final_model.get_layer('deploy').output)
    
    # predict on the NewCollege dataset
    names = os.listdir(imgs_path)
    num = len(names)
    for i in range(0, num):
        keyframe = mpimg.imread(imgs_path+'/'+names[i])
        descr = predict_on_keyframe(keyframe, deploy_model)
        descr = descr / np.linalg.norm(descr)   # normalize the descriptor
        np.save(preds_path+'/'+names[i][0:6]+'.npy', descr)
        if i%5 == 0:
            print('Save prediction results for %s'%(imgs_path+'/'+names[i]))

    return


def class2_score(x1, x2):
    '''
    Given two HOG vectors, calculate the inner product between them.
    Params:
        x1, x2: the normalized descriptors, with shape 1*936, 1 is num_imgs
    Output:
        score: similarity between x1 and x2
    '''
    score = np.dot(x1,x2.T)
    return score


def solve_confusion_matrix():
    '''
    Compute the confusion matrix based on CNN output descriptors
    '''
    
    global preds_path
    global confus_mat_path
    global N
    global thresh
    
    # descriptors
    files = os.listdir(preds_path)
    N = len(files)
    X = np.zeros((N,936),dtype=np.float32)
    for i in range(0,N):
        X[i,:] = np.load(preds_path+'/'+files[i])
    
    # upper triangular confusion matrix
    Mat = np.zeros((N,N), dtype=np.float32)
    for i in range(0,N):
        for j in range(i,N):
            x1 = X[i,:]
            x2 = X[j,:]
            Mat[i][j] = class2_score(x1,x2)
    print('Finsh computing the lower triangular part.')
    
    # fill in the lower traingular part
    for i in range(0,N):
        for j in range(0,i):
            Mat[i][j] = Mat[j][i]
    print('Map the lower triangular part to the upper traingular part.')
    
    
    Mat = thresh_matrix(Mat, thresh)
    np.save(confus_mat_path, Mat)
    
    return 
    

def thresh_matrix(mat, low):
    '''
    Set all entries in the input matrix smaller than 'low' to 0,
    which create larger spacing among entry values after normalization.
    Params:
        mat - the input matrix to apply thresholding
        low - entry smaller than 'low' will be set to zero
    '''
    ans = np.copy(mat)
    rows, cols = mat.shape
    for i in range(0, rows):
        for j in range(0, cols):
            if ans[i][j] < low:
                ans[i][j] = low
                
    return ans


def normalize_matrix(mat):
    normalized_mat = np.copy(mat)
    normalized_mat = cv2.normalize(mat, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return normalized_mat


def visualize_matrix(mat1, mat2):
    '''
    Plot the ground truth confusion matrix and computed confusion matrix.
    Remember to normalize the confusion matrix before visualization, also
    takes only the lower triangular part.
    Params:
        mat1: ground truth
        mat2: confusion matrix
    '''
    
    global color_map
    global compare_path
    
    f, axs = plt.subplots(1, 2, figsize=(10, 10), squeeze=False)
    f.tight_layout()
    f.subplots_adjust(hspace = 0.2, wspace = 0.1)
    axs = axs.ravel()
    
    axs[0].imshow(mat1, cmap=color_map)
    axs[0].set_title('Ground Truth', fontsize=20)
    axs[1].imshow(mat2, cmap=color_map)
    axs[1].set_title('Our Results', fontsize=20)
    
    f.savefig(compare_path, bbox_inches='tight')
    
def plot_precision_recall():
    '''
    plot the precision recall curve
    '''
    
    global confus_mat_path
    global groundtruth_path
    global prec_recall_path
    
    # groundtruth confusion matrix
    groundtruth = mpimg.imread(groundtruth_path)
    
    # load confusion matrix, normalize it, and take the lower triangular
    confus_mat = np.load(confus_mat_path)
    confus_mat = normalize_matrix(confus_mat)
    confus_mat = np.tril(confus_mat, -1)
    
    prec_recall_curve = []
    for thresh in np.arange(0.0, 1.05, 0.001):
        # precision: fraction of retrieved instances that are relevant
        # recall: fraction of relevant instances that are retrieved
        true_positives = (confus_mat > thresh) & (groundtruth == 1)
        all_positives = (confus_mat > thresh)

        try:
            precision = float(np.sum(true_positives)) / np.sum(all_positives)
            recall = float(np.sum(true_positives)) / np.sum(groundtruth == 1)

            prec_recall_curve.append([thresh, precision, recall])
        except:
            break
        
    prec_recall_curve = np.array(prec_recall_curve)
    
    #plt.plot(prec_recall_curve[:, 2], prec_recall_curve[:, 1])
    fig = plt.figure()
    fig.set_figheight(7)
    fig.set_figwidth(10)
    plt.plot(prec_recall_curve[:, 2], prec_recall_curve[:, 1])
    
    '''
    for thresh, prec, rec in prec_recall_curve[5::5]:
        plt.annotate(
            str(round(thresh * 1000) / 1000.0),
            xy=(prec, rec),
            xytext=(8, 8),
            textcoords='offset points')
    '''
    
    plt.ylabel('Precision', fontsize=20)
    plt.xlabel('Recall', fontsize=20)
    plt.title('Kitti Sequence 06 Results',fontsize=28)
    plt.tight_layout()
    plt.legend()
    plt.savefig(prec_recall_path, bbox_inches='tight')
    return

if __name__ == '__main__':
    
    ########## Parameters ##########
    data_path = 'E:/UnderRoot/my_dataset/KITTI_gray/sequences'
    seq = '06'
    thresh = 0.9999
    imgs_path = data_path + '/'+ seq + '/image_0'
    preds_path = data_path + '/'+ seq + '/predictions'
    model_path = 'E:/JuWorkDir/568_project/calc_model_6Million.h5'
    
    # results and evaluation
    confus_mat_path = data_path + '/' + seq + '/Confus_mat.npy'
    prec_recall_path = data_path + '/' + seq + '/prec_recall.png'
    compare_path = data_path + '/' + seq + '/comparation.png'
    groundtruth_path = 'E:/UnderRoot/my_dataset/KITTI_gray/loop_closure_groundtruth/'+ seq+ '/matrix' + seq+ '.png'
      
    
    
    # number of keyframes in the sequence, figured out in solve_confusion_matrix()
    N = 0   
    
    color_map = 'Purples'
    ########## Perform Test Here ##########
    
    
    # generate descriptor by pretrained CNN
    
    #predict_on_KITTI()
    
    # solve confusion matrix and groundtruth 
    solve_confusion_matrix()
    confus_mat = np.load(confus_mat_path)
    
    groundtruth = mpimg.imread(groundtruth_path)
    
    visualize_matrix(groundtruth, normalize_matrix(confus_mat))
    
    #plot_precision_recall()