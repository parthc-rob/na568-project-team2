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
