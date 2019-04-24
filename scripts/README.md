## This folder contains code for the project:
* prec_recall_figure.py run our trained model on the KITTI gray sequence and generate the confusion matrix and precision recall curve
* sim_score_hist.py run our trained model on the CampusLoopDataset and plot the similarity score histogram
* train.py, move.py and psedoDatasetGen.py are used in model training.

### Run prec_recall_figure.py
Please download KITTI grayscale images from http://www.cvlibs.net/datasets/kitti/eval_odometry.php and KITTI loop closure groud truth 
from http://www.robesafe.com/personal/roberto.arroyo/downloads.html (in section KITTI Odometry: Loop Closure Ground-Truth). Please 
download model file calc_model_6Million.h5 in master branch folder *models*. And AcrossChannelNorm.py in master branch folder *scripts*. From code lines 235~272 in prec_recall_figure.py modify the path variables:
* **put AcrossChannelNorm.py in the same folder of prec_recall_figure.py, this is a customized layer used by the network**
* set variable **data_path** as **YOUR_PATH_for_KITTI_FOLDER/dataset/sequences**
* create an empty folder named **predictions** under the directory **YOUR_PATH_for_KITTI__FOLDER/dataset/sequences/06**, this is the folder to store the descriptors computed by our network
* set variable **model_path** to **YOUR_PATH_for_h5modelfile/calc_model_6Million.h5** 
* set variable **groundtruth_path** to **YOUR_PATH_for_GROUNDTRUTH_FOLDER/06/matrix06.png*
* sequence besides 06 could also be tested, but some sequences don't have groundtruth provided. You also need to modify above path variables according to the sequence index you want to test

### Run sim_score_hist.py
Please download the CampusLoopDataset from https://github.com/rpng/calc/tree/master/TrainAndTest/test_data, unzip the package you will get two folders **live** and **memory**. In the same directory, create two new empty folders **livehogs** and **memoryhogs**, which are used to store descriptors computed by our model. 
From code line 61~66 in sim_score_hist.py modify the path variables:
* set variable **base** to **YOUR_PATH_for_DATASET/CampusLoopDataset/**
* set variable **model_path** to **YOUR_PATH_for_h5modelfile/calc_model_6Million.h5**
From code line 143~146 in sim_score_hist.py modify the path variables:
* set variable **livediscr_path** to **YOUR_PATH_for_DATASET/CampusLoopDataset/livehogs** 
* set variable **memorydiscr_path** to **YOUR_PATH_for_DATASET/CampusLoopDataset/memoryhogs** 

### Files we use to train our model
Plase download the Place365 dataset http://places2.csail.mit.edu/download.html. We choose the section **Data of Places365-Challenge 2016 small images 256x256, the 108GB training images and 501MB validation images**. After download, unzip the validation images, you will get a folder called **val_256** which contains 36500 raw validation jpg images. Unzip the training images, you will get a folder called **data_256**. This folder is much complex, with many different-level subfolders, and a total of 8 million images. So **move.py** is used to move all images in different subfolders into the same folder. In move.py 

**We use unsupervised learning here, we don't have a ground truth.** Then use **pseudoDatasetgen.py** to generate a psedo label for each training image and validation image, and also generate train images and validation images from raw images. You also need to modify path 
variables here.
Finally, use train.py to train the model. You will need to modify the path you save model and paths for train folder, trainlabels folder, val folder and vallabels folder.
