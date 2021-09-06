from time import sleep

import rospy
from geometry_msgs.msg import Point

rospy.init_node('test_chassis')

pub = rospy.Publisher("/client/chassis_slam/point", Point, queue_size=30)

goal_point = Point(x=-0.65588299604, y=8.25853904999, z=0.994025686939)
original_point = Point(x=2.08599934321, y=5.11935406091, z=0.703878971997)

pub.publish(goal_point)
