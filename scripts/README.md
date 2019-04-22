## This folder contains code for the project:
* trainvalSplit.py operates on a folder containing RGB images and separate it into two RGB images folders (*rawtrain* & *rawval*)
* run pseudoDatasetGen.py on folder *rawtrain* to get folders *train* and *trainlabels*
* run pseudoDatasetgen.py on folder *rawval* to get folders *val* and *vallables*
* set the path of folders *train, trainlabels, val, vallabels* in train.py to train the network.
* for test: check_match.py and test_on_kitti.py run tests 
