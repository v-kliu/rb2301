import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.INFO) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  

class LidarNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("LidarNode")
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to lidar
        self.is_simulation = None

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360: 
            self.is_simulation = True
        else:
            self.is_simulation = False

class ObstacleCourseNode(Node):
    '''Node to navigate obstacle course, using pose from either gazebo odometer or optitrack and lidar scan data'''
    def __init__(self, is_simulation:bool=True):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation

        if is_simulation:
            self.max_translate_velocity = 1.4
            self.goal_coordinates = np.array((5.2, -2.6))
        else:
            self.max_translate_velocity = 0.3 # Please keep this in place; 0.3m/s is more than fast enough 
            self.goal_coordinates = np.array((5.2, -2.6))

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.map_sub = self.create_subscription( 
                PoseStamped,
                '/vrpn_mocap/bingda_003/pose',
                self.optitrack_callback, 
                qos_profile
                )
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node       
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.pose = None
        self.last_scan = None


        # self.detection_thresholds = [0.13, 0.17, 0.15, 0.17] # self boundaries GAZEBO [0.18, 0.20, 0.24, 0.20] # self boundaries GAZEBO
        self.detection_thresholds = [0.15, 0.15, 0.24, 0.15]
        self.rightDeadlock = False
        self.rightDlPose = (0.0, 0.0)
        self.rightVel = 0.1

        self.leftDeadlock = False
        self.leftDlPose = (0.0, 0.0)
        self.leftVel = 0.1

        self.forwardVel = 0.1
        self.backVel = 0.1

        self.rightDeadlock = False
        self.leftDeadlock = False
        self.frontDeadlock = False
        self.prevRightDeadlock = False
            
        self.clearedRightDeadlock = False
        self.forwardAfterClearRight = False
        self.leftAfterRightDeadlock = False
        

        self.end = False
        self.getFrequency = False
        self.pendStart = False
        self.pendPastStart = False
        self.freqCalc = 0.00


        self.target = 0.0
        self.movingFirst = False
        self.movingSecond = False
        self.finishedFirst = False
        self.finishedSecond = False

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        if len(msg.ranges) <= 360:
            self.last_scan = np.array(msg.ranges)
        else:
            self.last_scan = np.array(msg.ranges)[::2] 

    def yaw_from_quaternion(self, q):
        '''Returns yaw angle (in rad) for orientation based on given quaternion input q'''
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def optitrack_callback(self, msg:PoseStamped):
        '''Callback to calculate 2D pose info from Optitrack node. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        x, y = msg.pose.position.x, msg.pose.position.y
        heading = np.rad2deg(self.yaw_from_quaternion(msg.pose.orientation))
        self.pose = np.array((x,y,heading))
        return self.pose

    def odometer_callback(self, msg):
        '''Callback to calculate 2D pose info from Gazebo odomoter. Pose info includes x and y coordinates, as well as heading in degrees.
        This callback will run everytime the rclpy executor spins'''
        latest_pose_msg = msg.pose.pose 
        heading = np.rad2deg(self.yaw_from_quaternion(latest_pose_msg.orientation))
        self.pose = np.array((latest_pose_msg.position.x, latest_pose_msg.position.y, heading))
        return self.pose

    def move_2D(self, x:float=0.0, y:float=0.0, turn:float=0.0):
        '''Publishes a Twist message to ROS to move a robot. Inputs are x and y linear velocities, as well as turn (z-axis yaw) angular velocity.'''
        twist_msg = Twist()
        x = np.clip(x, -self.max_translate_velocity, self.max_translate_velocity)
        y = np.clip(y, -self.max_translate_velocity, self.max_translate_velocity)
        turn = np.clip(turn, -self.max_translate_velocity*2, self.max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)



    def getMinDistanceInRange(self, offset, leftRangeOfView, rightRangeOfView):
        lid = self.last_scan
        currMin = lid[offset]
        linear_distances = []

        for leftAngle in range(1, leftRangeOfView):
            # use angle to calcualute
            l = lid[offset + leftAngle]
            x_offset = l * math.cos(math.radians(leftAngle))
            linear_distances.append(x_offset)

        for rightAngle in range(1, rightRangeOfView):
            l = lid[offset - rightAngle]
            x_offset = l * math.cos(math.radians(rightAngle))
            linear_distances.append(x_offset)

        # find min alue
        min_value = min(linear_distances)
        return min_value
    

    def pid(self):
        print("moving to waypoint", self.target, "current x is", self.pose[0])
        # errorX = self.nextX - self.currMapX
        # errorY = self.nextY - self.currMapY
        errorX = self.target - self.pose[0]


        print("errors", errorX)
        # if abs(errorX) <= 0.5 and abs(errorY) <= 0.5:
        moveX = 2 * errorX
        print("moving with", moveX)
        self.move_2D(moveX, 0.0, 0.0)

    def timer_callback(self):
        if self.last_scan is None:
            return # Does not run if the laser message is not received.
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            print("No pose detected")
            return # Does not run if no pose received from Odom or Optitrack
            
        elif np.linalg.norm(self.pose[:2] - self.goal_coordinates) < 0.05: # If distance to goal is less than 0.05m, consider goal reached and exit
            self.get_logger().info("Goal reached! Exiting script")
            raise SystemExit
        
        ###### INSERT CODE HERE ######
        self.get_logger().info(f"Pose: {self.pose}")
        # self.move_2D(0.1)

        ''' GAZEBO
        # create direction array first
        directions = np.zeros(4)
        directions[0] = self.getMinDistanceInRange(0, 40) # forward
        directions[2] = self.getMinDistanceInRange(180, 40) # back

        # --- NEW LOGIC for left/right scan range ---
        if (self.checkpoint_two and not self.checkpoint_three):
            # We are in Corridor 3, approaching Checkpoint 3 (the gate)
            # Use narrow 5-degree view to detect gate posts
            directions[1] = self.getMinDistanceInRange(90, 5) # left
            directions[3] = self.getMinDistanceInRange(270, 5) # right
        elif (self.checkpoint_three):
            # We are past Checkpoint 3, in the pendulum area.
            # Use narrow 5-degree view to avoid hitting side pendulum posts
            directions[1] = self.getMinDistanceInRange(90, 5) # left
            directions[3] = self.getMinDistanceInRange(270, 5) # right
        else:
            # We are in Corridor 1 or 2 (default)
            directions[1] = self.getMinDistanceInRange(90, 50) # left
            directions[3] = self.getMinDistanceInRange(270, 50) # right
        '''

        # # SIM
        # directions = np.zeros(4)
        # directions[0] = self.getMinDistanceInRange(0, 45, 45) # forward
        # directions[1] = self.getMinDistanceInRange(90, 58, 35) # left
        # directions[2] = self.getMinDistanceInRange(180, 35, 35) # back
        # directions[3] = self.getMinDistanceInRange(270, 35, 58) # right

        # REAL LIFE
        directions = np.zeros(4)
        directions[0] = self.getMinDistanceInRange(0, 20, 0) # forward
        directions[1] = self.getMinDistanceInRange(90, 62, 15) # left
        directions[2] = self.getMinDistanceInRange(180, 35, 35) # back
        directions[3] = self.getMinDistanceInRange(270, 15, 62) # right

        # --- END NEW LOGGING ---
        self.get_logger().error(f"Pose: {self.pose} and Directions: [{directions[0]}, {directions[1]}, {directions[2]}, {directions[3]}")


        # [0.3, 0.2, inf, 0.2] OLD THRESHOLDS from CA1
        blocked_info = ""
        # print debugging line
        if (directions[0] > self.detection_thresholds[0]) : blocked_info += "FRONT NOT BLOCKED, "
        else : blocked_info += "FRONT BLOCKED"
        blocked_info += str(directions[0])
        # if (directions[3] > self.detection_thresholds[3]) : blocked_info += "RIGHT NOT BLOCKED, "
        # else : blocked_info += "RIGHT BLOCKED (" + str(directions[3]) + "), "
        # if (directions[1] > self.detection_thresholds[1]) : blocked_info += "LEFT NOT BLOCKED, "
        # else: blocked_info += "LEFT BLOCKED (" + str(directions[1]) + "), "
        # if (directions[2] > self.detection_thresholds[2]) : blocked_info += "BACK NOT BLOCKED"
        # else : blocked_info += "BACK BLOCKED (" + str(directions[2]) + "), "

        self.get_logger().info(blocked_info)
        ###########
        if not self.end: #we havent started frequency tracking
            print("not done")
            if not self.finishedFirst: #prepare for frequency calculation
                print("Working on first bound")
                if not self.pendStart: # havent encountered pendulum passing since the time we set getFreuqency as True
                    #check if we encounter pendulum blocking
                    print("havent detected first pendulum")
                    if directions[0] < 0.25: #first encounter of first pendulum blocking robot
                        self.pendStart = True #trigger frequency calculation
                        print("detected first pendulum")
                elif not self.movingFirst:
                    print("not yet moving first pendulum")
                    if directions[0] > 0.25: #pendulum not reached back
                        self.target = self.pose[0] + 0.6
                        self.movingFirst = True
                        print("triggering moving past first pendulum")
                elif abs(self.pose[0] - self.target) >= 0.03 and directions[0] > self.detection_thresholds[0]: #not yet within desired threshold past first bound
                    #we have already ensured pendulum moved past robot and immediately triggered movement (self.movingFirst)
                    self.pid() #trigger movement past first bound
                    print("in progress - moving past first pendulum")
                else: #we have started pendulum, triggered movement, and made it past first bound within acceptable threshold (0.03)
                    print("sufficiently past first pendulum")
                    self.finishedFirst = True
                    self.pendStart = False
                    self.move_2D(0.0, 0.0, 0.0)
            elif not self.finishedSecond: #prepare for frequency calculation
                print("Working on second bound")
                if not self.pendStart: # havent encountered pendulum passing since the time we set getFreuqency as True
                    #check if we encounter pendulum blocking
                    print("havent detected second pendulum")
                    if directions[0] < 0.25: #first encounter of second pendulum blocking robot
                        self.pendStart = True #trigger frequency calculation
                        print("detected second pendulum")
                elif not self.movingSecond:
                    print("not yet moving first pendulum")
                    if directions[0] > 0.25: #pendulum not reached back
                        self.target = self.pose[0] + 0.6
                        self.movingSecond = True
                        print("triggering moving past second pendulum")
                elif abs(self.pose[0] - self.target) >= 0.03 and directions[0] > self.detection_thresholds[0]: #not yet within desired threshold past first bound
                    #we have already ensured pendulum moved past robot and immediately triggered movement (self.movingFirst)
                    self.pid() #trigger movement past first bound
                    print("in progress - moving past second pendulum")
                else: #we have started pendulum, triggered movement, and made it past second bound within acceptable threshold (0.03)
                    print("sufficiently past second pendulum")
                    self.finishedSecond = True
                    self.move_2D(0.0, 0.0, 0.0)
            else: #havent triggered frequency calculation
                self.end = True
        else:
            # print("obtained frequency:", self.freqCalc)
            print("done with all pendulums")
        
        # if not self.end: #we havent started frequency tracking
        #     print("not done")
        #     if self.getFrequency: #prepare for frequency calculation
        #         if not self.pendStart: # havent encountered pendulum passing since the time we set getFreuqency as True
        #             #check if we encounter pendulum blocking
        #             if directions[0] < self.detection_thresholds[0]: #first encounter of pendulum blocking robot
        #                 self.pendStart = True #trigger frequency calculation
        #         # elif not self.pendPastStart: #started but pendulum not past start
        #         else: #we have already encountered pendulum, so continue frequency calculation
        #             if directions[0] > self.detection_thresholds[0]: #pendulum not reached back
        #                 if not self.pendPastStart: 
        #                     #account for the case when previously there is delay from first encountering pendulum till when it swings past robot
        #                     self.pendPastStart = True
        #                 self.freqCalc += 0.05
        #             else: #pendulum blocking
        #                 if not self.pendPastStart: #pendulum still blocking robot (i.e. havent fully swung past robot)
        #                     self.freqCalc += 0.05
        #                 else: #pendulum has started, gone past start, and now reached back
        #                     #log frequency
        #                     self.getFrequency = False #we no longer want to calculate frequency
        #                     self.end = True # we do not want to trigger any more checks for frequency
        #     else: #havent triggered frequency calculation
        #         #check if pendulum does not exist infront
        #         if directions[0] > self.detection_thresholds[0]:
        #             #pendulum not in front of robot,
        #             #so we prepare to start frequency calculation
        #             self.getFrequency = True #this ensures the next time pendulum passes robot, we trigger frequency calculation
        # else:
        #     print("obtained frequency:", self.freqCalc)
            







        ###########
        ###### INSERT CODE HERE ######

        # if (directions[3] > self.detection_thresholds[3] and not self.rightDeadlock and not self.leftAfterRightDeadlock):
        #     self.frontDeadlock = False
        #     self.move_2D(0.0, -self.rightVel)
        #     print("moving right")
        #     if self.prevRightDeadlock: #we have moved forward along rightdeadlock but now we have passed it
        #         self.clearedRightDeadlock = True #keep track that we have cleared rightdeadlock
        #         #so that we can move backwards if we encounter front deadlock in subsequent rightdeadlock
        #     # else:
        #     #     self.clearedRightDeadlock = False
        #     # self.prevRightDeadlock = False
        #     self.forwardAfterClearRight = False
        # elif (directions[0] > self.detection_thresholds[0] and not self.frontDeadlock): 
        #     print("moving forward")
        #     #right blocked / deadlocked, but front clear and no known deadlock
        #     self.rightDeadlock = False
        #     self.leftDeadlock = False
        #     self.prevRightDeadlock = True
            
        #     if self.clearedRightDeadlock:
        #         self.forwardAfterClearRight = True
        #     # self.move_2D(self.forwardVel, 0.0)
        #     if self.leftAfterRightDeadlock:
        #         self.move_2D(0.5, 0.0)
        #     else:
        #         self.move_2D(self.forwardVel, 0.0)
        #     self.leftAfterRightDeadlock = False
        #     self.clearedRightDeadlock = False
        # else: 
        #     #right blocked / deadlocked and front blocked
        #     self.rightDeadlock = True
        #     self.frontDeadlock = True
        #     if self.clearedRightDeadlock or self.forwardAfterClearRight: 
        #         #since right is blocked / deadlocked and front is also blocked and we know we have cleared an earlier rightdeadlock
        #         #we can now backtrack
        #         # self.rightDeadlock = False
        #         if directions[2] > self.detection_thresholds[2]: #back is clear
        #             print("moving back after rightdeadlock")
        #             #keep forwardDeadlock as True
        #             #uncheck rightDeadlock so that we can check right while moving back
        #             self.move_2D(-self.backVel, 0.0)
        #         elif directions[1] > self.detection_thresholds[1] and not self.leftDeadlock:
        #             print("moving left after rightdeadlock")
        #             self.move_2D(0.0, self.leftVel)
        #             self.frontDeadlock = False
        #         else:
        #             self.move_2D(0.0, 0.0)
        #     else:
        #         if directions[1] > self.detection_thresholds[1] and not self.leftDeadlock:
        #             print("moving left normal")
        #             self.frontDeadlock = False
        #             self.forwardAfterClearRight = False
        #             self.leftAfterRightDeadlock = True

        #             #only disable rightdeadlock after we detect right blocked after right
        #             self.move_2D(0.0, self.leftVel)
        #         elif directions[2] > self.detection_thresholds[2]: #back is clear
        #             print("moving back normal")
        #             self.leftDeadlock = True
        #             #keep forwardDeadlock as True
        #             #uncheck rightDeadlock so that we can check right while moving back
        #             self.move_2D(-self.backVel, 0.0)
        #         else:
        #             self.move_2D(0.0, 0.0)
            
            



                

def main(args=None):
    print("Starting obstacle course")
    rclpy.init(args=args)

    # Use lidar scan length to determine if in simulation or not
    lidar = LidarNode()
    while lidar.is_simulation is None:
        rclpy.spin_once(lidar)
    is_simulation = lidar.is_simulation
    lidar.destroy_node()

    # Start spinning the obstacle node and only stop once SystemExit error is raised within the node callback
    obstacle = ObstacleCourseNode(is_simulation)
    try:
        rclpy.spin(obstacle)
    except SystemExit:
        print("Shutting down")
    obstacle.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()