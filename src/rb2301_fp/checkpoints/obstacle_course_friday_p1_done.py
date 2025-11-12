import numpy as np
import rclpy
import math
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from rclpy.qos import ReliabilityPolicy, QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(2, suppress=True, threshold=np.inf) # Print numpy arrays to specified d.p., suppress scientific notation and do not truncate
set_logger_level("obstaclecourse", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  

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
    def __init__(self, is_simulation:bool=False):
        super().__init__('obstaclecourse')
        self.get_logger().info("Starting ObstacleCourseNode")

        self.is_simulation = is_simulation

        if is_simulation:
            self.max_translate_velocity = 1.4
            self.goal_coordinates = np.array((5.2, -2.6))
        else:
            self.max_translate_velocity = 0.075 # Please keep this in place; 0.3m/s is more than fast enough 
            self.goal_coordinates = np.array((5.2, -2.6))

        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # Subscribe to LiDAR scan data

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.map_sub = self.create_subscription( 
                PoseStamped,
                '/vrpn_mocap/bingda_006/pose',
                self.optitrack_callback, 
                qos_profile
                )
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node       
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.last_scan = None
        self.pose = None
        self.move_speed = 0.3
        self.detection_thresholds = [0.105, 0.13, 0.18, 0.13] # self boundaries GAZEBO [0.18, 0.20, 0.24, 0.20] # self boundaries GAZEBO
        self.gate_detection_threshold = 0.35 # gate boundaries

        # field variable to keep track of permanant blockage
        self.blocked_second_priority_forward = False
        self.blocked_third_priority_forward = False
        self.blocked_fourth_priority_forward = False

        self.blocked_second_priority_right = False
        self.blocked_third_priority_right = False
        self.blocked_fourth_priority_right = False

        self.checkpoint_one = False
        self.checkpoint_two = False
        self.checkpoint_three = False

        self.swing_count = 0

        # NEW PENDULUM STATE MACHINE
        self.pendulum_state = "APPROACHING_GATE_1" # "APPROACHING_GATE_1", "TIMING_GATE_1_WAIT_FOR_AWAY", "TIMING_GATE_1_WAIT_FOR_RETURN", "WAITING_FOR_GAP_1", "PASSING_GATE_1", "TIMING_GATE_2_WAIT_FOR_AWAY", "TIMING_GATE_2_WAIT_FOR_RETURN", "WAITING_FOR_GAP_2", "PASSING_GATE_2", "FINAL_RUN", "DONE"
        self.pendulum_timer = 0
        self.pendulum_first_hit_time = 0
        self.pendulum_period_ticks = 0
        self.dynamic_move_speed = self.move_speed
        self.pendulum_hit_threshold = 0.18 # User-specified
        self.gate_pass_start_time = 0

        self.pendulum_start = False

        self.current_state = "NAV_FORWARD" 
        self.intended_corridor = "NAV_FORWARD"

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

    # gets the minimum distance in range of values
    def getMinDistanceInRange(self, offset, leftRangeOfView, rightRangeOfView):
        lid = self.last_scan
        linear_distances = []

        for leftAngle in range(1, leftRangeOfView):
            # use angle to calcualute
            l = lid[offset + leftAngle]
            if (np.isinf(l)) : continue
            x_offset = l * math.cos(math.radians(leftAngle))
            linear_distances.append(x_offset)

        for rightAngle in range(1, rightRangeOfView):
            l = lid[offset - rightAngle]
            if (np.isinf(l)) : continue
            x_offset = l * math.cos(math.radians(rightAngle))
            linear_distances.append(x_offset)

        # find min alue
        min_value = min(linear_distances)
        return min_value
    
    # starts in the middle of the gate
    def gate_logic(self, directions):
        directions[0] = self.getMinDistanceInRange(0, 50, 50) # forward
        directions[1] = self.getMinDistanceInRange(90, 65, 15) # left
        directions[2] = self.getMinDistanceInRange(180, 35, 35) # back
        directions[3] = self.getMinDistanceInRange(270, 15, 60) # right
        # if left is not blocked, prioritize go left
        if (not self.pendulum_start and directions[1] > self.detection_thresholds[1]):
            self.move_2D(0, self.move_speed, 0.0)
            self.get_logger().debug("STATE: GATE_LOGIC - GO LEFT )")
        else:
            # left is blocked and back is not blocked, start new gate logic
            if (not self.pendulum_start and directions[2] > self.detection_thresholds[2]):
                self.move_2D(self.move_speed, 0.0, 0.0)
                self.get_logger().debug("STATE: GATE_LOGIC - GO FORWARD )")
            else: # if left is blocked and back is blocked run forward, always run this
                self.pendulum_start = True
                # RUN GATE LOGIC
                
            self.move_2D(0.0, 0.0, 0.0)
            self.get_logger().debug("STATE: GATE_LOGIC - LEFT AND FRONT BLOCKED")

    # if left is not blocked
    # if left is blocked
        # if back is not blocked
        # else go forward

    # anytime going forward, prioritize going forward, then right, then left, then all the way to the right, then go back looking for right
    def navigate_forward_logic(self, directions):
        # forward moving logic

        # --- NEW TOP-LEVEL CHECK ---
        # If we are in Corridor 3 (past checkpoint_two) AND we've passed the gate (checkpoint_three),
        # run the PENDULUM logic exclusively.
        # if (self.checkpoint_two and self.checkpoint_three): 
        
        # --- END OF NEW TOP-LEVEL CHECK ---


        # --- COMBINED OBSTACLE AVOIDANCE LOGIC (for Corridor 1 AND Corridor 3 Pre-Gate) ---
        # if forward is greater than threshold go forward
        if (directions[0] > self.detection_thresholds[0] and not self.blocked_third_priority_forward):
            
            # This logic runs for P1 (Move Forward)
            self.move_2D(self.move_speed, 0.0, 0.0)
            
            # Since we successfully moved forward (or are about to), reset flags
            self.blocked_second_priority_forward = False
            self.blocked_third_priority_forward = False
            self.blocked_fourth_priority_forward = False
            
            self.get_logger().debug("STATE: NAV_FORWARD - Moving Forward")
            
        # forward is blocked, if haven't explored right (second priority) and not blocked go right
        elif (directions[3] > self.detection_thresholds[3] and not self.blocked_second_priority_forward):
            self.move_2D(0, -self.move_speed, 0.0)
            self.get_logger().debug("STATE: NAV_FORWARD - Moving Right (P2)")
        # forward and right is blocked, if haven't explored left (third priority) and not blocked go left
        elif (directions[1] > self.detection_thresholds[1] and not self.blocked_third_priority_forward):
            self.blocked_second_priority_forward = True
            self.move_2D(0, self.move_speed, 0.0)
            self.get_logger().debug("STATE: NAV_FORWARD - Moving Left (P3)")
        # blocked forward right left, need to backtrack somewhere, go all the way right and then go back looking for right
        else: 
            self.blocked_third_priority_forward = True
            # need to go back right
            if (directions[3] > self.detection_thresholds[3] and not self.blocked_fourth_priority_forward):
                self.move_2D(0, -self.move_speed, 0.0)
                self.get_logger().debug("STATE: NAV_FORWARD - Moving Right (P4)")
            else:
                self.blocked_fourth_priority_forward = True
                self.get_logger().debug("STATE: NAV_FORWARD - BLOCKED! Handing off to BACKTRACK_FROM_FORWARD state.")
                
                # reset flags
                self.blocked_second_priority_forward = False
                self.blocked_third_priority_forward = False
                self.blocked_fourth_priority_forward = False

                # set new state
                self.current_state = "BACKTRACK_FROM_FORWARD"
            
    # anytime going right, prioritize going right, then up, then down, then all the way up, then go back up looking for left
    def navigate_right_logic(self, directions):
        # right moving logic

        # if right is greater than threshold go right
        if (directions[3] > self.detection_thresholds[3] and not self.blocked_third_priority_right):
            self.move_2D(0.0, -self.move_speed, 0.0)
            self.blocked_second_priority_right = False
            self.blocked_third_priority_right = False
            self.blocked_fourth_priority_right = False
            self.get_logger().debug("STATE: NAV_RIGHT - Moving Right")
            
        # right is blocked, if haven't explored up (second priority) and not blocked go up
        elif (directions[0] > self.detection_thresholds[0] and not self.blocked_second_priority_right):
            self.move_2D(self.move_speed, 0.0, 0.0)
            self.get_logger().debug("STATE: NAV_RIGHT - Moving Forward (P2)")
        # right and up is blocked, if haven't explored down (third priority) and not blocked go down
        elif (directions[2] > self.detection_thresholds[2] and not self.blocked_third_priority_right):
            self.blocked_second_priority_right = True
            self.move_2D(-self.move_speed, 0.0, 0.0)
            self.get_logger().debug("STATE: NAV_RIGHT - Moving Down (P3)")
        # blocked right up down, need to backtrack somewhere, go all the way up and then go left
        else: 
            self.blocked_third_priority_right = True
            # need to go back up
            if (directions[0] > self.detection_thresholds[0] and not self.blocked_fourth_priority_right):
                self.move_2D(self.move_speed, 0.0, 0.0)
                self.get_logger().debug("STATE: NAV_RIGHT - Moving Forward (P4)")
            else:
                self.blocked_fourth_priority_right = True
                self.get_logger().debug("STATE: NAV_RIGHT - BLOCKED! Handing off to BACKTRACK_FROM_RIGHT state.")
                
                # reset flags for this state
                self.blocked_second_priority_right = False
                self.blocked_third_priority_right = False
                self.blocked_fourth_priority_right = False
                
                # set new state
                self.current_state = "BACKTRACK_FROM_RIGHT"

    def backtrack_from_forward_logic(self, directions):
        self.get_logger().debug("STATE: BACKTRACK_FROM_FORWARD - Trying to find Right path")
        # P1: try right (goal)
        if (directions[3] > self.detection_thresholds[3]):
            self.move_2D(0, -self.move_speed, 0.0)
            self.get_logger().debug("STATE: BACKTRACK_FROM_FORWARD - Found Right! Returning to normal nav.")
            self.current_state = self.intended_corridor # Go back to normal
        # P2: try back (escape)
        elif (directions[2] > self.detection_thresholds[2]):
            self.move_2D(-self.move_speed, 0.0, 0.0) 
            self.get_logger().debug("STATE: BACKTRACK_FROM_FORWARD - Right blocked, trying Back")
        # P3: try left (last resort)
        else:
            self.move_2D(0.0, self.move_speed, 0.0) 
            self.get_logger().debug("STATE: BACKTRACK_FROM_FORWARD - Right/Back blocked, trying Left")

    def backtrack_from_right_logic(self, directions):
        self.get_logger().debug("STATE: BACKTRACK_FROM_RIGHT - Trying to find Forward path")
        # P1: try forward (goal)
        if (directions[0] > self.detection_thresholds[0]):
            self.move_2D(self.move_speed, 0.0, 0.0)
            self.get_logger().debug("STATE: BACKTRACK_FROM_RIGHT - Found Forward! Returning to normal nav.")
            self.current_state = self.intended_corridor # Go back to normal
        # P2: try left (escape)
        elif (directions[1] > self.detection_thresholds[1]):
            self.move_2D(0.0, self.move_speed, 0.0) 
            self.get_logger().debug("STATE: BACKTRACK_FROM_RIGHT - Forward blocked, trying Left")
        # P3: try down (last resort)
        else:
            self.move_2D(-self.move_speed, 0.0, 0.0) 
            self.get_logger().debug("STATE: BACKTRACK_FROM_RIGHT - Forward/Left blocked, trying Down")


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

        # REAL LIFE
        directions = np.zeros(4)
        directions[0] = self.getMinDistanceInRange(0, 50, 50) # forward
        directions[1] = self.getMinDistanceInRange(90, 60, 15) # left
        directions[2] = self.getMinDistanceInRange(180, 35, 35) # back
        directions[3] = self.getMinDistanceInRange(270, 15, 60) # right

        # --- NEW ERROR LOGGING FOR CHECKPOINTS ---
        gate_status = "NOT PASSED"
        if self.checkpoint_three:
            gate_status = "PASSED"

        if not self.checkpoint_one:
            self.get_logger().error("STATUS: On Checkpoint 1. Gate Status: NOT PASSED")
        elif not self.checkpoint_two:
            self.get_logger().error("STATUS: On Checkpoint 2. Gate Status: NOT PASSED")
        else:
            self.get_logger().error(f"STATUS: On Checkpoint 3. Gate Status: {gate_status}")
        # --- END NEW LOGGING ---

        self.get_logger().error(f"Pose: {self.pose} and Directions: [{directions[0]}, {directions[1]}, {directions[2]}, {directions[3]}")

        # [0.3, 0.2, inf, 0.2] OLD THRESHOLDS from CA1
        blocked_info = ""
        # print debugging line
        if (directions[0] > self.detection_thresholds[0]) : blocked_info += "FRONT NOT BLOCKED, "
        else : blocked_info += "FRONT BLOCKED (" + str(directions[0]) + "), "
        if (directions[3] > self.detection_thresholds[3]) : blocked_info += "RIGHT NOT BLOCKED, "
        else : blocked_info += "RIGHT BLOCKED (" + str(directions[3]) + "), "
        if (directions[1] > self.detection_thresholds[1]) : blocked_info += "LEFT NOT BLOCKED, "
        else: blocked_info += "LEFT BLOCKED (" + str(directions[1]) + "), "
        if (directions[2] > self.detection_thresholds[2]) : blocked_info += "BACK NOT BLOCKED"
        else : blocked_info += "BACK BLOCKED (" + str(directions[2]) + "), "

        self.get_logger().debug(blocked_info)

        # '''
        # INTENDED
        # if pose is within forward part of zig zag, prioritize up
        # TRIGGER LINE 1 : y is 1.2 (slightl later becuase circle can spawn in middle -> should not optimzie for gazebo)
            # REAL LIFE 1 : y is 0.8
        # TRIGGER LINE 2 : x -1.8 (more aggresive should go forward no buggy circle to worry about)
            # REAL LIFE 2 : X - 0.6 
        trigger_lines = [1.30, -1.3, 1.9]
        # else, prioritize right
        if (not self.checkpoint_one):
            # we are in Corridor 1
            self.intended_corridor = "NAV_FORWARD"
            if (self.pose[0] > trigger_lines[0]): # Check for exit condition
                self.checkpoint_one = True
                self.get_logger().info("--- COMPLETED CORRIDOR 1, ENTERING CORRIDOR 2 ---")
        elif (not self.checkpoint_two):
            # we are in Corridor 2
            self.intended_corridor = "NAV_RIGHT"
            if (self.pose[1] < trigger_lines[1]): # Check for exit condition
                self.checkpoint_two = True
                self.get_logger().info("--- COMPLETED CORRIDOR 2, ENTERING CORRIDOR 3 ---")
        elif (not self.checkpoint_three):
            # we are in Corridor 3
            self.intended_corridor = "NAV_FORWARD"  
            if (self.pose[0] > trigger_lines[2]): # Check for exit condition
                self.checkpoint_three = True
                self.get_logger().info("--- COMPLETED CORRIDOR 3, ENTERING CORRIDOR 4 ---")   
        else: # gate logic
            self.intended_corridor = "GATE LOGIC"

        # 2. If we aren't backtracking, follow the intended corridor
        if "BACKTRACK" not in self.current_state:
            self.current_state = self.intended_corridor

        # 3. STATE MACHINE DISPATCHER
        if self.current_state == "NAV_FORWARD":
            self.navigate_forward_logic(directions)
        elif self.current_state == "NAV_RIGHT":
            self.navigate_right_logic(directions)
        elif self.current_state == "BACKTRACK_FROM_FORWARD":
            self.backtrack_from_forward_logic(directions)
        elif self.current_state == "BACKTRACK_FROM_RIGHT":
            self.backtrack_from_right_logic(directions)
        elif self.current_state == "GATE LOGIC":
            self.gate_logic(directions)
        
        self.get_logger().debug(str(self.pose[2]))

        # again want to constanatly move up and right
        # '''

        ###### INSERT CODE HERE ######
                

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




