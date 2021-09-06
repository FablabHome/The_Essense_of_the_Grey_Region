#!/usr/bin/env python3
import argparse
import rospy
from geometry_msgs.msg import PoseStamped
import sys


def main(args, goal_pub):
    msg = PoseStamped()
    x, y, z, w = args['point']
    msg.header.frame_id = 'map'
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.z = z
    msg.pose.orientation.w = w

    while rospy.get_param('/status_monitor/status_code') != 0:
        goal_pub.publish(msg)

    if args['wait_until_end']:
        while rospy.get_param('/status_monitor/status_code') != 3:
            continue


if __name__ == '__main__':
    rospy.init_node('go_to_point', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--point', nargs=4,
                        type=float,
                        required=True,
                        help='point for robot to go, (x, y, z. w)')
    parser.add_argument('--wait-until-end', action='store_true',
                        help="Wait until the slam has end")
    args = vars(parser.parse_args())
    goal_pub = rospy.Publisher(
        '/move_base_simple/goal',
        PoseStamped,
        queue_size=1
    )

    try:
        main(args, goal_pub)
        sys.exit(0)
    except Exception as e:
        print(f'Program ended due to: {e}')
        sys.exit(1)

