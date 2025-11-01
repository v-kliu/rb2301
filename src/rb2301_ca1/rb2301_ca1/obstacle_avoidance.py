
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.logging import set_logger_level, LoggingSeverity
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

np.set_printoptions(
    2, suppress=True
)  # Print numpy arrays to specified d.p. and suppress scientific notation (e.g. 1e-5)

max_translate_velocity = 0.4 # Can be implemented as parameter
max_turn_velocity = max_translate_velocity * 2 # Can be implemented as parameter
set_logger_level("obstacle_avoidance", level=LoggingSeverity.DEBUG) # Configure to either LoggingSeverity.INFO or LoggingSeverity.DEBUG  

class ObstacleAvoidanceNode(Node):
    def __init__(self):
        """Node constructor"""
        super().__init__("obstacle_avoidance")
        self.get_logger().info("Starting Obstacle Avoidance")

        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)  # Publish to cmd_vel node
        self.sub_scan = self.create_subscription(LaserScan, "scan", self.sub_scan_callback, 2) # The subscriber to the Lidar ranges.
        self.last_scan = None # Copied laser scan message

        self.timer = self.create_timer(0.05, self.timer_callback)  # Runs at 20Hz. Can be changed.

        # field variable to keep track of permanant blockage
        self.blocked_left = False

    def move_2D(self, x: float = 0.0, y: float = 0.0, turn: float = 0.0):
        """Publishes a twist command to move in 2D space. +ve x is forwards, +ve y is left, and +ve turn is anticlockwise"""
        twist_msg = Twist()
        # Please keep the max velocity caps to protect the irl robots
        twist_msg.linear.x, twist_msg.linear.y, twist_msg.linear.z = (
            min(float(x), max_translate_velocity),
            min(float(y), max_translate_velocity),
            0.0,
        )
        twist_msg.angular.x, twist_msg.angular.y, twist_msg.angular.z = (
            0.0,
            0.0,
            min(float(turn), max_turn_velocity),
        )
        self.pub_cmd_vel.publish(twist_msg)

    def sub_scan_callback(self, msg):
        """Scan subscriber"""
        self.last_scan = np.array(msg.ranges)[::20] # Slices the 721 scan array to return only 36 scans. Feel free to edit

    def timer_callback(self):
        """Controller loop"""

        if self.last_scan is None:
            return # Does not run if the laser message is not received.
        
        ######################## MODIFY CODE HERE ########################
        self.get_logger().debug(str(self.last_scan))
        
        # get the lidar array first
        lid = self.last_scan
        # forward is : 0 +- 30 degrees
        forward_results = [lid[0], lid[1], lid[2], lid[34], lid[35], lid[3], lid[33]]
        # left is : 90 degrees +- 20 degrees
        left_results = [lid[9], lid[10], lid[11], lid[8], lid[7]]
        # back is : 180 +- 20 degrees
        back_results = [lid[18], lid[17], lid[16], lid[19], lid[20]]
        # right is : 270 +- 20 degrees
        right_results = [lid[27], lid[28], lid[29], lid[26], lid[25]]
        

        # create direction array first
        directions = np.zeros(4)
        # forward
        directions[0] = min(forward_results)
        # left
        directions[1] = min(left_results)
        # back
        directions[2] = min(back_results)
        # right
        directions[3] = min(right_results)

        # forward moving logic
        # if forward is infinity (clear) or greater than 0.3
        if (directions[0] > 0.3):
            self.move_2D(0.3, 0.0, 0.0)
            self.blocked_left = False
            self.get_logger().debug("MOVE FORWARD")
        # if left is clear and greater than 2 go left
        elif (directions[1] > 0.2 and not self.blocked_left):
            self.move_2D(0, 0.15, 0.0)
            self.get_logger().debug("MOVE LEFT")
        # if right is clear and greater than 2 go right
        elif (directions[3] > 0.2):
            self.blocked_left = True
            self.move_2D(0, -0.15, 0.0)
            self.get_logger().debug("MOVE RIGHT")
        # if back is clear go back
        elif (np.isinf(directions[2])):
            self.move_2D(-0.2, 0.0, 0.0)
            self.get_logger().debug("MOVE BACK")
        # else, somehow we are trapped and breakS
        else:
            self.get_logger().debug("FAILURE")

        self.get_logger().debug(f"Directions: F:{directions[0]} L:{directions[1]} B:{directions[2]} R:{directions[3]}")

        ######################## MODIFY CODE HERE ########################


def main(args=None):
    rclpy.init(args=args)
    obstacle_avoidance_node = ObstacleAvoidanceNode()
    rclpy.spin(obstacle_avoidance_node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
