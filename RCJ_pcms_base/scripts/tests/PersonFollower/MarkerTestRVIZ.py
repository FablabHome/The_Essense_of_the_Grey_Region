#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray

rospy.init_node('test_marker')
pub = rospy.Publisher(
    '/visualization_marker_array',
    MarkerArray,
    queue_size=1
)
rospy.sleep(2)
marker_array = MarkerArray()

marker = Marker()
marker.header.frame_id = "map"
marker.header.stamp = rospy.Time()
marker.ns = "my_namespace"
marker.id = 1
marker.type = Marker.SPHERE
marker.lifetime = rospy.Duration(0)
marker.action = Marker.ADD
marker.pose.position.x = 0.0
marker.pose.position.y = 0.0
marker.pose.position.z = 0.0
marker.pose.orientation.x = 0.0
marker.pose.orientation.y = 0.0
marker.pose.orientation.z = 0.0
marker.pose.orientation.w = 0.0
marker.scale.x = 0.3
marker.scale.y = 0.3
marker.scale.z = 0.3
marker.color.r = 0.0
marker.color.g = 0.0
marker.color.b = 1.0
marker.color.a = 1.0

marker_array.markers.append(marker)

pub.publish(marker_array)
rospy.sleep(5)
