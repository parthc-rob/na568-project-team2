## **Active SLAM**

### Definition
A decision making problem requires the robot to choose between maximize explored areas(frontier action) and minimized SLAM uncertainty(loop
closing action).

### Structure
A possible structure is to integrate an occupancy grid map based SLAM and a motion choosing mechanism.
#### SLAM
We may use a Rao-Blackwellized particle filter SLAM for its robustness on nonlinear models. And there are two possible motion choosing 
mechanism to choose between.
#### Active loop closing with topological mapping


