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

        self.last_scan = None
        self.pose = None
        self.move_speed = 0.3
        self.detection_thresholds = [0.18, 0.20, 0.24, 0.20]

        # field variable to keep track of permanant blockage
        self.blocked_second_priority_forward = False
        self.blocked_third_priority_forward = False
        self.blocked_fourth_priority_forward = False

        self.blocked_second_priority_right = False
        self.blocked_third_priority_right = False
        self.blocked_fourth_priority_right = False

        self.trigger_one = False
        self.trigger_two = False

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
    def getMinDistanceInRange(self, offset, rangeOfView):
        lid = self.last_scan
        currMin = lid[offset]

        # incorporate angles from [-rangeOfView + offset, offset + rangeOfView]
        for angle in range(1, rangeOfView):
            currMin = min(currMin, lid[offset + angle])
            currMin = min(currMin, lid[offset - angle])

        return currMin
    
    # anytime going forward, prioritize going forward, then right, then left, then all the way to the right, then go back looking for right
    def navigate_forward_logic(self, directions):
        # forward moving logic

        # if forward is greater than threshold go forward
        if (directions[0] > self.detection_thresholds[0] and not self.blocked_third_priority_forward):
            self.move_2D(self.move_speed, 0.0, 0.0)
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
        self.get_logger().info(f"Pose: {self.pose}")
        # self.move_2D(0.1)

        # create direction array first
        directions = np.zeros(4)
        directions[0] = self.getMinDistanceInRange(0, 40) # forward
        directions[1] = self.getMinDistanceInRange(90, 50) # left
        directions[2] = self.getMinDistanceInRange(180, 40) # back
        directions[3] = self.getMinDistanceInRange(270, 50) # right


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

        # INTENDED
        # if pose is within forward part of zig zag, prioritize up
        # TRIGGER LINE 1 : y is 1.2 (slightl later becuase circle can spawn in middle -> should not optimzie for gazebo)
        # TRIGGER LINE 2 : x -1.8 (more aggresive should go forward no buggy circle to worry about)
        # else, prioritize right
        if (not self.trigger_one):
            # we are in Corridor 1
            self.intended_corridor = "NAV_FORWARD"
            if (self.pose[0] > 1.2): # Check for exit condition
                self.trigger_one = True
                self.get_logger().info("--- COMPLETED CORRIDOR 1, ENTERING CORRIDOR 2 ---")
        elif (not self.trigger_two):
            # we are in Corridor 2
            self.intended_corridor = "NAV_RIGHT"
            if (self.pose[1] < -1.8): # Check for exit condition
                self.trigger_two = True
                self.get_logger().info("--- COMPLETED CORRIDOR 2, ENTERING CORRIDOR 3 ---")
        else:
            # we are in Corridor 3
            self.intended_corridor = "NAV_FORWARD"     

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


        self.get_logger().debug(str(self.pose[2]))

        # again want to constanatly move up and right

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