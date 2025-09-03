import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose

class Recorder(Node):
    def __init__(self):
        super().__init__('recorder')
        
        self.f = open('data.txt', 'w')
        self.subscription = self.create_subscription(Pose, '/turtle1/pose', self.pose_callback, 10)

    def pose_callback(self, msg):
        # tab seperate x, y, and theta
        self.f.write(f'{msg.x}\t{msg.y}\t{msg.theta}\n')

def main(args=None):
    rclpy.init(args=args)
    node = Recorder()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()