import rospy
from geometry_msgs.msg import PoseStamped

rospy.init_node('test_navigation', anonymous=True)

pub = rospy.Publisher(
    '/move_base_simple/goal',
    PoseStamped,
    queue_size=1
)

msg = PoseStamped()
msg.header.frame_id = 'map'
msg.pose.position.x = -4.82
msg.pose.position.y = 0.827
msg.pose.orientation.z = 0.0688
msg.pose.orientation.w = 0.9999762

found = False

while not rospy.is_shutdown():
    pub.publish(msg)
