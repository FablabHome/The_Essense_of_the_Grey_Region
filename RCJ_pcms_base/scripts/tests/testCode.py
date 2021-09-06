#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

rospy.init_node('test_code', anonymous=True)
pub = rospy.Publisher(
    '/main_prog/code',
    String,
    queue_size=1
)

rospy.sleep(2)

pub.publish('beta')
