#!/usr/bin/env python3
import sys

import rospy
from std_msgs.msg import String


def main(code):
    global pub
    pub.publish(code)


if __name__ == '__main__':
    rospy.init_node('code', anonymous=True)

    code = sys.argv[1]

    pub = rospy.Publisher(
        '/main_prog/code',
        String,
        queue_size=1
    )
    rospy.sleep(.5)

    try:
        main(code)
        sys.exit(0)
    except Exception as e:
        print(f'Program ended due to: {e}')
        sys.exit(1)
