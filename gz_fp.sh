#!/bin/bash
source install/setup.bash
#ros2 launch rb2301_gz fp_gazebo.launch.py x:=0.0 y:=0.0
# ros2 launch rb2301_gz fp_gazebo.launch.py x:=-0.25 y:=0.0 # more correct launch area
ros2 launch rb2301_gz fp_gazebo.launch.py x:=2.6 y:=-2.6 # start at checkpoint two, go fowward 

# ros2 launch rb2301_gz fp_gazebo.launch.py x:=3.6 y:=-2.6 # Start past the gate

./kill.sh # kills any lingering Gazebo