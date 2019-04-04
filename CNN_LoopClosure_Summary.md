## **About the Deep Learning Part for Loop Closure Detecting**

Tasks we can do:

* Reproduce the main results of the paper "Lightweight Unsupervised Deep Loop Closure"
* Connect with ORB-SLAM2 backend and apply the network architecture on other datasets
* Rewrite the code, including: image preprocessing, dataset modification, network training(from caffe to keras, pytorch, etc.) 
* Possibly slightly modify the network architecture and finetune parameters

[//]: # (Image References)

[image1]: test_images/cnn_loopClosure_fig1.jpg "HOG_match"
[image2]: test_images/cnn_loopClosure_fig2.jpg "viewpoint_warp"
[image3]: test_images/cnn_loopClosure_fig3.jpg "network_semantic"


### Paper Review

The work detects visual loop closure by using a combination of convolutional neural network and nereast neighbour search. It ustilizes the
fact that the image after performing large viewpoint and illumination changes still has very similar HOG features (Histogram of Gradient).

![alt text][image1]

