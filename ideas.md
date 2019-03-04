# Three Potential idea with Source code

## CNN-SVO

#### CNN-SVO: Improving the Mapping in Semi-Direct Visual Odometry Using Single-Image Depth Prediction

Abstract— Reliable feature correspondence between frames is a critical step in visual odometry (VO) and visual simultaneous localization and mapping (V-SLAM) algorithms. In comparison with existing VO and V-SLAM algorithms, semi-direct visual odometry (SVO) has two main advantages that lead to state-of-the-art frame rate camera motion estimation: direct pixel correspondence and efﬁcient implementation of probabilistic mapping method. This paper improves the SVO mapping by initializing the mean and the variance of the depth at a feature location according to the depth prediction from a singleimage depth prediction network. By signiﬁcantly reducing the depth uncertainty of the initialized map point (i.e., small variance centred about the depth prediction), the beneﬁts are twofold: reliable feature correspondence between views and fast convergence to the true depth in order to create new map points. We evaluate our method with two outdoor datasets: KITTI dataset and Oxford Robotcar dataset. The experimental results indicate that the improved SVO mapping results in increased robustness and camera tracking accuracy.

https://github.com/yan99033/CNN-SVO

> Improved depth estimation using neural network. Converge faster and robust to illumination variation.



## RGB-iD-SLAM

#### RGBiD-SLAM for Accurate Real-time Localisation and 3D Mapping 

In this paper we present a complete SLAM system for RGB-D cameras, namely RGB-iD SLAM. The presented approach is a dense direct SLAM method with the main characteristic of working with the depth maps in inverse depth parametrisation for the routines of dense alignment or keyframe fusion. The system consists in 2 CPU threads working in parallel, which share the use of the GPU for dense alignment and keyframe fusion routines. The ﬁrst thread is a front-end operating at frame rate, which processes every incoming frame from the RGB-D sensor to compute the incremental odometry and integrate it in a keyframe which is changed periodically following a covisibility-based strategy. The second thread is a back-end which receives keyframes from the front-end. This thread is in charge of segmenting the keyframes based on their structure, describing them using Bags of Words, trying to ﬁnd potential loop closures with previous keyframes, and in such case perform pose-graph optimisation for trajectory correction. In addition, our system allows is able to compute the odometry both with unregistered and registered depth maps, allowing to use customised calibrations of the RGB-D sensor. As a consequence in the paper we also propose a detailed calibration pipeline to compute customised calibrations for particular RGB-D cameras. The experiments with our approach in the TUM RGB-D benchmark datasets show results superior in accuracy to the state-of-the-art in many of the sequences. The code has been made available on-line for research purposes.

https://github.com/dangut/RGBiD-SLAM





## Unsupervised learning

#### Unsupervised Learning of Monocular Depth Estimation and Visual Odometry with Deep Feature Reconstruction

Despite learning based methods showing promising results in single view depth estimation and visual odometry, most existing approaches treat the tasks in a supervised manner. Recent approaches to single view depth estimation explore the possibility of learning without full supervision via minimizing photometric error. In this paper, we explore the use of stereo sequences for learning depth and visual odometry. The use of stereo sequences enables the use of both spatial (between left-right pairs) and temporal (forward backward) photometric warp error, and constrains the scene depth and camera motion to be in a common, real-world scale. At test time our framework is able to estimate single view depth and two-view odometry from a monocular sequence. We also show how we can improve on a standard photometric warp loss by considering a warp of deep features. We show through extensive experiments that: (i) jointly training for single view depth and visual odometry improves depth prediction because of the additional constraint imposed on depths and achieves competitive results for visual odometry; (ii) deep feature-based warping loss improves upon simple photometric warp loss for both single view depth estimation and visual odometry. Our method outperforms existing learning based methods on the KITTI driving dataset in both tasks. 

The source code is available at https://github.com/Huangying-Zhan/Depth-VO-Feat



# Semantic SLAM(not )



## Probabilistic Data Association for Semantic SLAM

* No deep learning involved 
* EM Estimation



## VSO: Visual Semantic Odometry



## Stereo Vision-based Semantic 3D Object and Ego-motion Tracking for Autonomous Driving





## Long-term Visual Localization using Semantically Segmented Images

