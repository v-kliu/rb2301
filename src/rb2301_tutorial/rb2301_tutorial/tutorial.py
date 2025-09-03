# Victor Liu - e1675208

# 1. IMPORT
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn

# NODE CLASS
class Tutorial(Node):
    
    # 2. NODE PROPERTIES
    counter = 0
    odom_msg = None
    turtle = None

    def __init__(self):
        super().__init__('tutorial')
        # 3. NODE CONSTRUCTOR
        self.odom_sub = self.create_subscription(Pose, '/turtle1/pose', self.odom_sub_callback, 10)
        self.odom_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.spawn_cli = self.create_client(Spawn, '/spawn')
        self.timer = self.create_timer(0.5, self.timer_callback)

    # 4. NODE CALLBACKS
    def timer_callback(self):
        # Spawn turtle only on init 
        if Tutorial.turtle is None:
            request = Spawn.Request()
            request.x = 1.0
            request.y = 1.0
            request.name = ''
            Tutorial.turtle = self.spawn_cli.call_async(request)

        # Should always run since Turtle either exists or was just created
        if Tutorial.turtle is not None:
            if Tutorial.turtle.done():
                print(f'Turtle name: {Tutorial.turtle.result().name}')

        # Increment counter
        Tutorial.counter += 1

        # Get time and print time and position
        time_now = self.get_clock().now().seconds_nanoseconds()
        print(f'Another .5 sec! sec: {time_now[0]}, nanosec: {time_now[1]} counter is at: {Tutorial.counter}')
        if Tutorial.odom_msg is not None:
            print(f'({Tutorial.odom_msg.x}, {Tutorial.odom_msg.y})')

        # Modify velocity and publish
        velocity = Twist()
        if Tutorial.counter % 2 == 1:
            velocity.linear.x = 1.0
        else:
            velocity.linear.x = -1.0
        self.odom_pub.publish(velocity)

    def odom_sub_callback(self, msg):
        Tutorial.odom_msg = msg

    # 5. HOW TO USE

# MAIN BOILER PLATE
def main(args=None):
    rclpy.init(args=args)
    node = Tutorial()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()