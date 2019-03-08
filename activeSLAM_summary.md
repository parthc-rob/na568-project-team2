## **Active SLAM**

### Definition
A decision making problem requires the robot to choose between maximize explored areas(frontier action) and minimized SLAM uncertainty(loop
closing action).

### Structure
A possible structure is to integrate an occupancy grid map based SLAM and a motion choosing mechanism.
#### SLAM
We may use a Rao-Blackwellized particle filter SLAM for its robustness on nonlinear models. And there are two possible motion choosing 
mechanism to choose between.
#### Old-fashioned node-based method
#### Selection among a finite set
* Select vantage points
* Compute the utility of applying a certain motion, such as information gain
* Carry out the selected motion and determine whether to terminate the task



### Reference
1. C. Stachniss, D. Hahnel, and W. Burgard, "Exploration with active loop-closing for FastSLAM", 2004 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (IEEE Cat. No.04CH37566). 
2. L. carlone, J. Du, M. Kaouk, B. Bona, and M. Indri, "Active SLAM and exploration with particle filters using Kullback-Leibler divergence", J. Intell. Robot. Syst., vol. 75, no. 2, pp. 291-311, 2014.


