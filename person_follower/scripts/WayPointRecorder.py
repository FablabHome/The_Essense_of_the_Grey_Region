#!/usr/bin/env python3
from typing import List

import numpy as np
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from home_robot_msgs.msg import PFWaypoints, PFWaypoint
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

from core.Nodes import Node


class WayPointRecorder(Node):
    # Waypoint recording parameters
    MAX_RECORDS = 60
    UPDATE_DURATION = 0.1

    # Camera specs
    FOV_H = 60
    FOV_V = 49.5

    # Cam/era W and H
    W = 640
    H = 480

    # Camera uplifted angle
    CAMERA_ANGLE = 17

    def __init__(self):
        super(WayPointRecorder, self).__init__('waypoint_recorder')

        self.waypoints: List[List] = []
        self.current_pose = self.last_pose = [0., 0., 0., 0.]
        self.update_duration = rospy.get_rostime() + rospy.Duration(WayPointRecorder.UPDATE_DURATION)
        self.log_file = open('/tmp/test.txt', 'w+')

        self.waypoints_pub = rospy.Publisher(
            '~waypoints',
            PFWaypoints,
            queue_size=1
        )
        self.visualization_pub = rospy.Publisher(
            '/visualization_marker_array',
            MarkerArray,
            queue_size=1
        )
        rospy.Subscriber(
            '/PFRHandler/fake_waypoint',
            PFWaypoint,
            self.callback,
            queue_size=1
        )
        rospy.Subscriber(
            '/amcl_pose',
            PoseWithCovarianceStamped,
            self.pose_callback,
            queue_size=1
        )

        rospy.set_param('~max_records', WayPointRecorder.MAX_RECORDS)
        rospy.set_param('~update_duration', WayPointRecorder.UPDATE_DURATION)

        self.main()

    def callback(self, point: PFWaypoint):
        if rospy.get_rostime() - self.update_duration >= rospy.Duration(0):
            # Reset the timer
            self.update_duration = rospy.get_rostime() + rospy.Duration(rospy.get_param('~update_duration'))

            # Convert waypoint to true x, y, z
            x, y, z = point.x, point.y, point.z
            real_x, real_y, real_z = self.convert_waypoint_to_real(x, y, z)
            real_x /= 1000
            real_y /= 1000
            real_z /= 1000

            # Record the waypoint
            WayPointRecorder.MAX_RECORDS = rospy.get_param('~max_records')
            self.waypoints.append([real_x, real_y, real_z])

            if not len(self.last_pose) == 0:
                for waypoint in self.waypoints[:len(self.waypoints) - 1]:
                    x, y, _, _ = self.current_pose
                    lx, ly, _, _ = self.last_pose

                    elx = x - lx
                    ely = y - ly

                    waypoint[0] += elx
                    waypoint[2] += ely

            text = str(self.waypoints)
            self.log_file.write(text[1:len(text) - 1] + '\n')
        else:
            return

        self.last_pose = self.current_pose

    def pose_callback(self, pose: PoseWithCovarianceStamped):
        x = pose.pose.pose.position.x
        y = pose.pose.pose.position.y
        z = pose.pose.pose.orientation.z
        w = pose.pose.pose.orientation.w
        self.current_pose = [x, y, z, w]

    def convert_waypoint_to_real(self, x, y, z):
        rad_h = self.angle_2_radian(WayPointRecorder.FOV_H / 2)
        rad_v = self.angle_2_radian(WayPointRecorder.FOV_V / 2)
        rad_cam_angle = self.angle_2_radian(WayPointRecorder.CAMERA_ANGLE)
        real_w = 2 * z * np.tan(rad_h)
        real_h = 2 * z * np.tan(rad_v)

        # Real x
        real_x = real_w * x / WayPointRecorder.W
        real_x -= (real_w / 2)

        # Real y
        reality_y = real_h * y / WayPointRecorder.H
        FE = z * np.sin(rad_cam_angle)
        GF = (0.5 * real_h - reality_y) * np.cos(rad_cam_angle)
        real_y = FE + GF

        # Real z
        OD = z * np.cos(rad_cam_angle)
        ED = GF * np.tan(rad_cam_angle)
        real_z = OD - ED

        return real_x, real_y, real_z

    @staticmethod
    def angle_2_radian(angle):
        return (np.pi * angle) / 180

    @staticmethod
    def into_marker(id, px, py, pz, qz, qw, r, g, b, a):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()
        marker.ns = "my_namespace"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.lifetime = rospy.Duration(0)
        marker.action = Marker.ADD
        marker.pose.position.x = px
        marker.pose.position.y = py
        marker.pose.position.z = pz
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

        return marker

    def main(self):
        while not rospy.is_shutdown():
            # Avoid overflowing waypoints according to ~max_records
            overflow_error = len(self.waypoints) - WayPointRecorder.MAX_RECORDS
            if overflow_error > 0:
                for _ in range(overflow_error):
                    self.waypoints.pop(0)

            marker_array = MarkerArray()
            for idx, waypoint in enumerate(self.waypoints):
                x, y, d = waypoint
                cx, cy, *czw = self.current_pose
                cr = euler_from_quaternion([0, 0, *czw])[2]

                alpha = np.arctan(x / d)
                beta = cr - alpha
                GE = d / np.cos(alpha)
                px = cx + GE * np.cos(beta)
                py = cy + GE * np.sin(beta)
                if idx == len(self.waypoints) - 1:
                    color = (0, 255, 0, 1)
                    pz = .5
                else:
                    color = (0, 0, 255, 1)
                    pz = .25

                marker = self.into_marker(idx, px, py, pz, 0, 0, *color)
                marker_array.markers.append(marker)

            serialized_waypoints = list(map(lambda pos: PFWaypoint(pos[0], pos[1], pos[2]), self.waypoints))
            self.visualization_pub.publish(marker_array)
            self.waypoints_pub.publish(PFWaypoints(serialized_waypoints))

    def reset(self):
        pass


if __name__ == '__main__':
    node = WayPointRecorder()
