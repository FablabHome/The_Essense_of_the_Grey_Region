import cv2 as cv
import numpy as np
import rospy
from home_robot_msgs.msg import PFWaypoints


def callback(wpts: PFWaypoints):
    global waypoints
    waypoints = wpts.waypoints


rospy.init_node('draw_waypoint')
rate = rospy.Rate(30)

waypoints = []
cm2pixel_ratio = 1 / 10

rospy.Subscriber(
    '/waypoint_recorder/waypoints',
    PFWaypoints,
    callback,
    queue_size=1
)

while not rospy.is_shutdown():
    board = np.zeros((480, 640, 3))
    for waypoint in waypoints:
        x = waypoint.x
        y = waypoint.y
        z = waypoint.z

        wpt_x = int(x * cm2pixel_ratio + 320)
        wpt_y = int(z * cm2pixel_ratio + 240)

        cv.circle(board, (wpt_x, wpt_y), 5, (32, 255, 0), -1)

    cv.imshow('map', board)
    key = cv.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break

    rate.sleep()

cv.destroyAllWindows()
