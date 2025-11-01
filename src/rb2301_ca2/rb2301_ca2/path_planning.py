# Imports 
import math
import numpy as np
import heapq
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity

from rclpy.qos import (
    ReliabilityPolicy,
    QoSProfile,
)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from PIL import Image
from geometry_msgs.msg import Twist

np.set_printoptions(
    2, suppress=True, threshold=np.inf
) 

# Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  
set_logger_level("waypoint", level=LoggingSeverity.DEBUG) 

# Variable to keep appropriate speed for simualation vs real lab
is_simulation = False 
if is_simulation:
    max_translate_velocity = 1.4
else:
    max_translate_velocity = 0.3 

occupancy_grid_resolution = 0.2

# Waypoints/goals!
sim_goal_list = [(3.2, 0.2), (2.4, -3.6), (-0.4, -3.8)]
sim_grid_start = (-1.0, -5.0)

irl_goal_list = [(2.1, -1.7), (2.3, -0.3), (1.5, -1.7), (0.3, -1.7)]
irl_grid_start = (0.1, -1.9)

# Main WaypointNode
class WaypointNode(Node):
    # ============ INITIALIZATION ============
    '''Node to calculate path and move robot towards given goal_coordinates, using pose info from either gazebo odometer or optitrack'''
    def __init__(self, map_array:np.array, goal_list:list, is_simulation:bool=True):
        super().__init__('waypoint')
        self.get_logger().info("Starting WaypointNode")

        self.is_simulation = is_simulation

        # Subscribe to the dynamic_pose topic from Gazebo that publishes ground-truth pose data
        if self.is_simulation:
            self.get_logger().info("STILL RUNNING SIMULATION")
            self.subscription = self.create_subscription(Odometry, 'odom', self.odometer_callback, 2)
        else:
            self.get_logger().info("SHOULD BE RUNNING")
            qos_profile = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)

            self.map_sub = self.create_subscription( 
                PoseStamped,
                '/vrpn_mocap/bingda_007/pose',
                self.optitrack_callback, 
                qos_profile
                )
            
            self.get_logger().info("CREATED SUBSCRIPTION TO ROBOT SUCCESSFULLY")
            
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10) # Publish to cmd_vel node       
        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        self.goal_list = goal_list
        self.map_array = map_array

        self.pose = None

        self.grid = Grid(self.map_array)

        self.found = False

        self.offset = (lambda : sim_grid_start if self.is_simulation else irl_grid_start)()
        # print(offset == sim_grid_start)
        # print(offset == irl_grid_start)
        self.goals = (lambda : sim_goal_list if self.is_simulation else irl_goal_list)()


        self.currPoseX = 0.0
        self.currPoseY = 0.0
        self.currMapX, self.currMapY = 0, 0
        self.poseRectX, self.poseRectY = 0.0, 0.0
        self.initial = True

        self.finished = False
        self.reachedGoal = True
        self.reachedWaypoint = True
        
        self.currGoalIndex = 0
        self.goalMapX, self.goalMapY = 0, 0

        self.nextX, self.nextY = 0, 0
        self.waypoints = []
        self.full_path_for_debug = []
        self.waypointIndex = 0
        self.currDir = None
        
        # HEADING SYSTEM: variables for heading-based navigation
        self.use_heading_control = True  # Set to False to use original strafing
        self.heading_aligned = False
        self.heading_tolerance = 2.0  # degrees
        self.position_tolerance = 0.04  # meters

    # ============ MAIN CONTROL LOOP ============
    def timer_callback(self):
        """Controller loop. Insert path planning and PID control logic here"""
        if self.pose is None:
            self.get_logger().info("IS SELF.POSE IS NULL")
            return # Does not run if no pose received from Odom or Optitrack
        self.get_logger().debug(f"Pose: {self.pose}")

        # self.get_logger().debug(f"OPTIMAL HEADING ERROR IS: {self.normalize_angle(90.0, self.pose[2])}")
        
        self.currPoseX, self.currPoseY = [round(i, 2) for i in self.pose[:2]]
        self.currMapX, self.currMapY = self.gridToIndex(self.offset, self.currPoseX, self.currPoseY)
        print("currMap", self.currMapX, self.currMapY)

        self.poseRectX, self.poseRectY = self.poseRectCoordinate()
        print("currRect", self.poseRectX, self.poseRectY)
        
        
        if self.finished:
            print("Done")
            self.move_2D(0, 0, 0)
        elif self.reachedGoal: #havent finished, but reached previous goal
            #so retrieve next goal target
            hasNextGoal = self.retrieveGoal()
            if hasNextGoal:
                print("next goal retrieved", self.goalMapX, self.goalMapY)

                # get raw_path of waypoints from astar function
                raw_path = self.astar((self.currMapX, self.currMapY), (self.goalMapX, self.goalMapY))
                self.full_path_for_debug = raw_path # Store the full list for logging
                #we take waypoints from index 1 onwards since the start point not needed
                # we assume we are already at the start point
                self.waypoints = raw_path[1:] # Use the sliced list for the controller

                self.get_logger().debug(f"NEW PATH CALCULATED: {len(self.waypoints)} waypoints: {self.waypoints}")

                self.reachedGoal = False
            else:
                print("finished all goals")
                self.finished = True
        
        else: #neither finished nor reached current goal
            print("not at goal")
            if self.reachedWaypoint: # if have reached a waypoint
                # check if our waypointIndex is out of bounds, meaning we have reached all waypoints
                if self.waypointIndex >= len(self.waypoints):  
                    # reached goal
                    print("reached goal")
                    self.reachedGoal = True
                    self.waypointIndex = 0
                    self.move_2D(0, 0, 0)
                else: # else we get waypointIndex which already stores the next waypoint (dont increment before)
                    print("getting next waypoint after", self.nextX, self.nextY)
                    (self.nextX, self.nextY) = self.waypoints[self.waypointIndex]
                    self.nextX, self.nextY = self.mapMidCoordinate(self.nextX, self.nextY)
                    self.reachedWaypoint = False
                    self.waypointIndex += 1
            else:
                self.get_logger().debug(f"Goal #{self.currGoalIndex}: Path: {self.full_path_for_debug} --> Moving to ({self.nextX:.2f}, {self.nextY:.2f})")
                # continue to current unreached waypoint
                if self.use_heading_control:
                    self.pid_with_heading()
                else:
                    self.pid()

    # ============ PATHFINDING ============
    # Runs at the start of each waypoint, maps out path plan for robot by running A*
    def astar(self, start:tuple=(0, 0), end:tuple=(0, 0)):
        shapeX, shapeY = np.shape(self.map_array)
        print(shapeX, shapeY)

        # exploredGrids contains all visited and pre-existing obstalces
        exploredGrids = np.copy(self.map_array)
        
        #store each element in open_list as (cost, current coordinates, parent coordinates, bias)
        open_list = [(0, start, start, 0)]
        print("Open list", open_list)
        closed_list = {}

        found = False
        while not found: 
            # get min cost node from heap O(log n) time complexity
            minDistNode = heapq.heappop(open_list)
            # print("Curr minDistNode and Target", minDistNode, end)
            currX, currY = minDistNode[1]
            # print("curr x and y", currX, currY)
            if (currX, currY) == end:
                found = True
            
            #adding to finished explored node list
            exploredGrids[currX][currY] = 1

            #to indicate current node is the start node
            #this is needed so bias calculation will not try to find direction for parent
            # if the current node is the start node (hence no parent node)
            root = False 

            #adding explored node to closed list
            if len(closed_list) == 0:
                closed_list[minDistNode[1]] = {
                    "curr": minDistNode[1],
                    "parent": minDistNode[1],
                    "dist": 0
                }
                root = True
            else:
                closed_list[minDistNode[1]] = {
                    "curr": minDistNode[1],
                    "parent": minDistNode[2],
                    "dist": closed_list[minDistNode[2]]["dist"] + 1 + minDistNode[3]
                    #new closed list distance is the sum of parent node distance + 1 + bias
                    #where bias > 0 means direction had changed (which we want to discourage)
                }
            # print("curr closed list", [[closed_list[item]["curr"], closed_list[item]["parent"]] for item in closed_list.keys()])
            
            #adding new nodes to explore 
            prev_dir = self.getDir(minDistNode[1], minDistNode[2])
            # print("CurrX, shapeX", currX, shapeX)
            # print("CurrY, shapeY", currY, shapeY)

            # if within bounds and haven't been explored, then explore - RIGHT NODE CASE
            if currX < shapeX - 1 and exploredGrids[currX + 1][currY] == 0:
                currNode, parentNode = (currX + 1, currY), (currX, currY)
                bias = self.getBias(currNode, parentNode, prev_dir, root)
                self.addToOpenList(currNode, parentNode, bias, open_list, closed_list, end)
                # print("added right node")
            
            # if within bounds and haven't been explored, then explore - BOTTOM NODE CASE
            if currY < shapeY - 1 and exploredGrids[currX][currY + 1] == 0: 
                    currNode, parentNode = (currX, currY + 1), (currX, currY)
                    bias = self.getBias(currNode, parentNode, prev_dir, root)
                    self.addToOpenList(currNode, parentNode, bias, open_list, closed_list, end)
                    # print("added bottom node")

            # if within bounds and haven't been explored, then explore - LEFT NODE CASE
            if currX > 0 and exploredGrids[currX - 1][currY] == 0:
                    currNode, parentNode = (currX - 1, currY), (currX, currY)
                    bias = self.getBias(currNode, parentNode, prev_dir, root)
                    self.addToOpenList(currNode, parentNode, bias, open_list, closed_list, end)
                    # print("added left node")

            # if within bounds and haven't been explored, then explore - TOP NODE CASE
            if currY > 0 and exploredGrids[currX][currY - 1] == 0: 
                    currNode, parentNode = (currX, currY - 1), (currX, currY)
                    bias = self.getBias(currNode, parentNode, prev_dir, root)
                    self.addToOpenList(currNode, parentNode, bias, open_list, closed_list, end)
                    # print("added top node")
            
        # print(closed_list, open_list)
        path = [end]
        waypoints = [end]
        curr_dir = self.getDir(end, end)
        prev_dir = self.getDir(end, end)
        curr = end
        child = end
        root = True
        while curr != start:
            item = closed_list[curr]
            child = curr
            curr = item["parent"]
            prev_dir = curr_dir
            curr_dir = self.getDir(curr, child)
            if root:
                root = False
            else:
                if curr_dir != prev_dir:
                    waypoints.append(child)
            
            path.append(curr)
        waypoints.append(start)
        waypoints = waypoints[::-1]
        # self.grid.draw_grid_map([start, end], list(closed_list.keys()))
        self.grid.draw_grid_map(waypoints, path)
        return waypoints
    
    def retrieveGoal(self):
        if self.currGoalIndex >= len(self.goals):
            return False
        # currPoseX, currPoseY = [round(i, 2) for i in self.pose[:2]]
        # print(self.currPoseX, self.currPoseY)
        # currMapX, currMapY = self.gridToIndex(self.offset, self.currPoseX, self.currPoseY)
        # print("currMap", self.currMapX, self.currMapY)

        goalGridX, goalGridY = self.goals[self.currGoalIndex]
        print("goal grid", goalGridX, goalGridY)
        self.goalMapX, self.goalMapY = self.gridToIndex(self.offset, goalGridX, goalGridY)
        print("goalMap", self.goalMapX, self.goalMapY)

        self.currGoalIndex += 1
        
        return True

    # ============ ROBOT CONTROL - PID ============
    def pid(self):
        print("moving to waypoint", self.nextX, self.nextY)
        # errorX = self.nextX - self.currMapX
        # errorY = self.nextY - self.currMapY
        errorX = self.nextX - self.poseRectX
        errorY = self.nextY - self.poseRectY


        print("errors", errorX, errorY)
        # if abs(errorX) <= 0.5 and abs(errorY) <= 0.5:
        if abs(errorX) < 0.1 and abs(errorY) < 0.1:
            print("reached waypoint")
            self.reachedWaypoint = True
        else:
            moveX = 0
            moveY = 0
            # if abs(errorX) >= 0.5:
            if abs(errorX) >= 0.1:
                moveX = 2 * errorX
            # if abs(errorY) >= 0.5:
            if abs(errorY) >= 0.1:
                moveY = 2 * errorY
            print("moving with", moveX, moveY)
            self.move_2D(moveX, moveY)
            # moveX = 2 * errorX
            # moveY = 2 * errorY
            # print("moving with", moveX, moveY)
            # self.move_2D(moveX, moveY)

    # HEADING SYSTEM: New PID controller with heading control
    def pid_with_heading(self):
        print("moving to waypoint", self.nextX, self.nextY)
        
        errorX = self.nextX - self.poseRectX
        errorY = self.nextY - self.poseRectY
        distance = math.sqrt(errorX**2 + errorY**2)
        
        print("errors", errorX, errorY, "distance:", distance)
        
        if distance < self.position_tolerance:
            print("reached waypoint")
            self.reachedWaypoint = True
            self.heading_aligned = False
            self.move_2D(0, 0, 0)
            return
        
        target_heading = self.calculate_target_heading(self.nextX, self.nextY)
        current_heading = self.pose[2]
        heading_error = self.normalize_angle(target_heading, current_heading)
        
        print(f"Current heading: {current_heading:.1f}°, Target: {target_heading:.1f}°, Error: {heading_error:.1f}°")
        
        # need to correct heading
        if abs(heading_error) > self.heading_tolerance:
            print("Rotating to align with target")
            self.heading_aligned = False
            turn_speed = np.clip(5 * math.radians(heading_error), -max_translate_velocity * 1.5, max_translate_velocity * 1.5)
            print(f"Sending turn command: {turn_speed}")  
            self.move_2D(0, 0, turn_speed)
        else: # already facing right heading, just move forward
            self.heading_aligned = True
            
            forward_speed = np.clip(4.0 * distance, 0, max_translate_velocity)
            print("Moving forward with speed", forward_speed)
            self.move_2D(forward_speed, 0, 0)

    # ============ HELPERS - coordinate conversion ============
    # To convert grid coordinate (bottom right indexed with an offset) to occupancy grid index
    # This is used when we want to move to the next goal (given in grid coordinates), 
    # then given input goal in grid coordinates we find the corresponding map index
    def gridToIndex(self, offset:tuple=(0.0, 0.0), gridX:float=0.0, gridY:float=0.0):
        return int(math.ceil((gridX - offset[0]) / 0.2)), int(math.ceil((gridY - offset[1]) / 0.2))

    def poseRectCoordinate(self):
        # return round(self.currPoseX - self.offset[0], 1), round(self.currPoseY - self.offset[1], 1)
        return self.currPoseX - self.offset[0], self.currPoseY - self.offset[1]
    
    
    def mapMidCoordinate(self, mapX, mapY):
        return mapX*0.2, mapY*0.2

    # ============ HELPERS - a* ============
    # #To convert pose coordinate (bottom left indexed) to occupancy grid index
    # #This is used when we want to convert current position (given in pose coordinates) 
    # # to corresponding map index
    # #making use of known column range to be 30
    # def poseToMap(self, offset:tuple=(0.0, 0.0), poseX:float=0.0, poseY:float=0.0):
    #     return poseX / 0.2, (poseY + offset[1]) / 0.2 (30 * 0.2)) / 0.2

    # Calculates Euclidean distance been current node and goal node
    def heuristic(self, currNode, endNode):
        return math.sqrt((currNode[0] - endNode[0]) ** 2 + (currNode[1] - endNode[1]) ** 2)
    
    # Calculates cardinal cardinal direction from parent node to current node, used in bias calculation
    def getDir(self, currNode, parentNode):
        if currNode[0] < parentNode[0]:
            return "u"
        elif currNode[0] > parentNode[0]:
            return "d"
        elif currNode[1] < parentNode[1]:
            return "l"
        else:
            return "r"
        
    # Bias that favors straight-line paths, adding 0.1 pentalty for turns
    def getBias(self, currNode, parentNode, prev_dir, root):
        if root:
            return 0
        curr_dir = self.getDir(currNode, parentNode)
        if curr_dir != prev_dir:
            return 0.1
        return 0
        
    # In the open list, evaluates a path to neighbor and updates it in Open List accordingly
    def addToOpenList(self, currNode, parentNode, bias, open_list, closed_list, end):
        #cost for new node given by distance from source to parent node 
        # + distance from parent node to new node (just 1 here, since neighbouring squares are same distance apart)
        # + bias (due to directional change)
        # + heuristic cost (through heuristic function)
        currCost = closed_list[parentNode]["dist"] + 1 + bias + self.heuristic(currNode, end)
        added = False #track if in the subsequent for loop we have found matching the node
        
        for i in range(len(open_list)):
            if open_list[i][1] == currNode: #currNode is already in open_list (now at index 1)
                if currCost < open_list[i][0]: #we have a better cost (now at index 0)
                    open_list.pop(i)
                    heapq.heappush(open_list, (currCost, currNode, parentNode, bias))
                added = True
                break
        #if no matching node was found, we havent updated the list, so we need to add
        if not added:
            heapq.heappush(open_list, (currCost, currNode, parentNode, bias))

    # ============ HELPERS - heading control ============
    # HEADING SYSTEM: Helper functions for heading-based navigation
    def normalize_angle(self, target_angle, curr_angle): 
        # if curr_angle > 180.0:
        #     curr_angle = 360.0 + curr_angle
        if curr_angle < 0.0:
            curr_angle = 360.0 + curr_angle

        if abs(target_angle - curr_angle) <= 180:
            return target_angle - curr_angle
        elif target_angle < curr_angle:
            return target_angle + 360 - curr_angle
        else: #target > current
            return -(curr_angle + 360 - target_angle)

    # based on deltas, calculate target heading
    def calculate_target_heading(self, target_x, target_y):
        
        dx = target_x - self.poseRectX
        dy = target_y - self.poseRectY
        target_heading = math.degrees(math.atan2(dy, dx))

        # if target_heading > 180.0:
        #     target_heading = 360.0 + target_heading
        print("targetx, targety", target_x, target_y)
        print("target heading before", target_heading)
        if target_heading < 0.0:
            target_heading = 360.0 + target_heading

        print("dx, dy, poseX, poseY", dx, dy, self.poseRectX, self.poseRectY)
        print("target heading after", target_heading)

        return target_heading
                
    # ============ ROS CALLBACKS & COMMUNICATION (provided so not touching LOL)============
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
        x = np.clip(x, -max_translate_velocity, max_translate_velocity)
        y = np.clip(y, -max_translate_velocity, max_translate_velocity)
        turn = np.clip(turn, -max_translate_velocity*2, max_translate_velocity*2)
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = float(x), float(y), 0.0
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = 0.0, 0.0, float(turn)
        self.publisher_.publish(twist_msg)

    def set_waypoints(self, waypoints:list):
        '''Set new waypoints when a goal has been reached'''
        self.goal_reached = False
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        
class Grid():
    '''
    Grid class to use with occupancy grid. Contains the following functions:
        check_grid_validity : Uses flood fill to check if there's a valid path from start to goal position
        draw_grid_map : Creates a colour image of the grid, as well as waypoints and full solution path if given.

    __init__ input Args:
        grid_array : 2D numpy array representing the occupancy grid
        starting_position : tuple of starting indices within the numpy array
        goal_position : tuple of goal indices within the numpy array. Works with negative indices as well
    '''
    def __init__(self, grid_array:np.array=np.array([]), starting_position:tuple=(0,0), goal_position:tuple=(-1,-1)):
        self.grid = grid_array
        self.shape = self.grid.shape
        self.starting_position = starting_position

        # If goal position given with negative index, need to convert to +ve
        if goal_position[0] < 0:
            goal_x = self.shape[0] + goal_position[0]
        else:
            goal_x = goal_position[0]
        if goal_position[1] < 0:
            goal_y = self.shape[1] + goal_position[1]
        else:
            goal_y = goal_position[1]   

        self.goal_position = (goal_x, goal_y)

    def check_grid_validity(self):
        '''Use flood fill to check if there's a viable path between start and goal positions'''
        grid = self.grid.copy()
        flood_stack = [(self.goal_position[0], self.goal_position[1])]
        while flood_stack:
            tile = flood_stack[0]
            del flood_stack[0]
            try:
                next_tile = (tile[0]+1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0]-1, tile[1])
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]+1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
            try:
                next_tile = (tile[0], tile[1]-1)
                if grid[next_tile] == 0:
                    grid[next_tile] = 1
                    flood_stack.append(next_tile)
            except:pass
        if grid[self.starting_position] == 1: # Means the flood is able to reach starting position from the ending position
            return True
        else:
            return False

    def draw_grid_map(self, waypoints:list=(), path:list=(), obstacle_threshold:float=50):
        '''Creates an image of the maze and path taken. Maze walls in blue, empty space in white, path taken in green and waypoints in red

        Args:
            waypoints : list (or other iterable) of tuple coordinates representing all the grid indices for the waypoints. Will be represented in red, takes precedence over path
            path : list (or other iterable) of tuple coordinates representing all the grid indices forming the solution path. Will be represented in green
            obstacle_threshold : Optional float to indicate threshhold for whether a grid is considered occupied. Not important for ca2             
        '''
        image_grid = np.ones((self.grid.shape[0],self.grid.shape[1],3), dtype=np.uint8)
        image_grid[self.grid <= obstacle_threshold] = (255,255,255)
        image_grid[self.grid > obstacle_threshold] = (0,0,255)

        for x, y in path:
            image_grid[x][y] = (0,255,0)

        for point in waypoints:
            image_grid[point] = (255,0,0)

        image_grid = np.flip(image_grid, axis=1)[::-1]
        img = Image.fromarray(image_grid, 'RGB')

        # Resize image
        base_width = 500
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), Image.Resampling.NEAREST)

        img.show()


def main(args=None):
    global is_simulation
    print("Starting path planning")
    rclpy.init(args=args)

    # Load the proper occupancy grid numpy array
    import os
    filepath = os.path.dirname(os.path.realpath(__file__))
    if is_simulation:
        map_array = np.load(filepath + '/ca2_sim_map.npy', allow_pickle=True)
        waypoint = WaypointNode(map_array, sim_goal_list, is_simulation)
    else:
        map_array = np.load(filepath + '/ca2_irl_map.npy', allow_pickle=True)
        waypoint = WaypointNode(map_array, irl_goal_list, is_simulation)

    # Start spinning the waypoint node and only stop once SystemExit error is raised within the node callback
    try:
        rclpy.spin(waypoint)
    except SystemExit:
        print("Shutting down")

    waypoint.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()