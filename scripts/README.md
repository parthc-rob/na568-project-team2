## This folder contains code for the project:
* prec_recall_figure.py run our trained model on the KITTI gray sequence and generate the confusion matrix and precision recall curve
* sim_score_hist.py run our trained model on the CampusLoopDataset and plot the similarity score histogram
* other files are used for training the network

### run prec_recall_figure.py
Please download KITTI grayscale images from http://www.cvlibs.net/datasets/kitti/eval_odometry.php and KITTI loop closure groud truth 
from http://www.robesafe.com/personal/roberto.arroyo/downloads.html (in section KITTI Odometry: Loop Closure Ground-Truth). Please 
download model file calc_model_6Million.h5 in master branch folder *model*s. And AcrossChannelNorm.py in master branch folder *scripts*. From code lines 235~272 in prec_recall_figure.py modify the path variables:
* put AcrossChannelNorm.py in the same folder of prec_recall_figure.py, this is a customized layer used by the network
* modify *data_path* to *your_path_store_KITTI_grayimage_folder/dataset/sequences*
* create an empty folder named *predictions* under *your_path_storing_KITTI_grayimage_folder/dataset/sequences/06*, this is the folder to 
store the descriptors computed by our network
* modify *model_path* to *your_path_store_h5modelfile/calc_model_6Million.h5* 
* modify *groundtruth_path* to *your_path_store_groundtruth_foler/06/matrix06.png*
* sequence other than 06 could also be tested, you need to modify path variables according to the sequence you want to test.

### run sim_score_hist.py
Please download the CampusLoopDataset from https://github.com/rpng/calc/tree/master/TrainAndTest/test_data, unzip the package and in the same directory storing folders *live* and *memory* create two new empty folders *livehogs* and *memoryhogs*, which are used to store 
descriptors computed by our model. 
From code line 61~66 in sim_score_hist.py modify the path variables:
* modify *base* to *your_path_store_campusloopdataset/CampusLoopDataset/*
* modify *model_path* to *your_path_store_h5modelfile/calc_model_6Million.h5* 
From code line 143~146 in sim_score_hist.py modify the path variables:
* modify *livediscr_path* to *your_path_store_campusloopdataset/CampusLoopDataset/livehogs* 
* modify *memorydiscr_path* to *your_path_store_campusloopdataset/CampusLoopDataset/memoryhogs* 

### files we use to train our model
The functionality of each file is briefly explained as follow. Plase download the Place365 dataset http://places2.csail.mit.edu/download.html. We choose the section *Data of Places365-Challenge 2016 small images 256*256*, the 108GB training images and 501MB validation images. After download, unzip the validation images, you will get a folder called val_256 which contains 36500 jpg. Unzip the training images, you will get a folder called data_256. This folder is much complex, with many different-level subfolders. So **move.py** is used to move all images in different subfolders into the same folder, this will facilitate the Keras data generator to parse images, you need to modify the path varibles here for your need. These are all raw images, and need to be handled preprocessing later. 
**We use unsupervised learning here, we don't have a ground truth.** Then use **pseudoDatasetgen.py** to generate a psedo label for each training image and validation image, and also generate train images and validation images from raw images. You also need to modify path 
variables here.
Finally, use train.py to train the model. You will need to modify the path you save model and paths for train folder, trainlabels folder, val folder and vallabels folder.
