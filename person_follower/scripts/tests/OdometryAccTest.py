import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion


def imu_callback(imu: Imu):
    global angular_accel
    q = [
        imu.orientation.x,
        imu.orientation.y,
        imu.orientation.z,
        imu.orientation.w
    ]
    _, _, angular_accel = euler_from_quaternion(q)


rospy.init_node('test_odemetry_acc')
angular_accel = .0

twist_pub = rospy.Publisher(
    '/mobile_base/commands/velocity',
    Twist,
    queue_size=1
)

rospy.Subscriber(
    '/mobile_base/sensors/imu_data',
    Imu,
    imu_callback,
    queue_size=1
)


timeout = rospy.get_rostime() + rospy.Duration(1.35344)
while rospy.get_rostime() - timeout <= rospy.Duration(0):
    twist = Twist()
    twist.linear.x = 0.5
    twist.angular.z = 3.14
    twist_pub.publish(twist)
