## **About the Deep Learning Part for Loop Closure Detecting**

Tasks we can do:

* Reproduce the main results of the paper "Lightweight Unsupervised Deep Loop Closure"
* Connect with ORB-SLAM2 backend and apply the network architecture on other datasets
* Rewrite the code, including: image preprocessing, dataset modification, network training(from caffe to keras, pytorch, etc.) 
* Possibly slightly modify the network architecture and finetune parameters

[//]: # (Image References)

[image1]: test_images/cnn_loopClosure_fig1.JPG "HOG_match"
[image2]: test_images/cnn_loopClosure_fig2.JPG "viewpoint_warp"
[image3]: test_images/cnn_loopClosure_fig3.JPG "network_semantic"


### Paper Review

The work detects visual loop closure by using a combination of convolutional neural network and nereast neighbour search. It ustilizes the
fact that the image after performing large viewpoint and illumination transformations still has very similar HOG features (Histogram of Gradient).

![alt text][image1]

For each single image, HOG is a very long 1D vector. The image shape is 120*160 and HOG shape is 1*3648 in the paper. In the training step, for each training image, do the folliwng steps:

* Convert the RGB image into grayscale, denotes img_1A
* Apply random viewpoint transformation on img_1A and store the result as img_1B.
![alt text][image2]
* Shuffle the pair (img_1A, img_1B). Pick one image from the pair to use as the training image, named as X1, and the other image in the pair is used to compute HOG vector, named X2.
* The HOG vector X2 is used as training "label" 
* Stack (X1,X2) of all training images as the training set

So the network aims to compute a HOG feature for each input image. In the test step, the network computes the HOG vector for the test image and uses NN search to judge whether this place has been seen before.

### Network Architecture
![alt text][image3]

The network is not very complex (depth within LeNet-5 and VGG-16, much simpler than Inception-V1).  

### Reference
1. Nate Merrill and Guoquan Huang, *Lightweight Unsupervised Deep Loop Closure*, https://arxiv.org/abs/1805.07703
