#!/usr/bin/env python3
from datetime import datetime

import cv2 as cv
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import TriggerResponse

from core.Nodes import Node


class FollowDatasetCollector(Node):
    def __init__(self, name, anonymous=False):
        super(FollowDatasetCollector, self).__init__(name, anonymous)
        self.srcframe = None
        self.bridge = CvBridge()

        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        self.video_writer = cv.VideoWriter(str(datetime.now()), fourcc, Node.ROS_RATE, (640, 480))

    def __camera_callback(self, img: CompressedImage):
        self.srcframe = self.bridge.compressed_imgmsg_to_cv2(img)

    def main(self):
        while not rospy.is_shutdown():
            frame = self.srcframe.copy()
            self.video_writer.write(frame)
            cv.imshow('frame', frame)
            cv.waitKey(1)
            self.rate.sleep()

    def reset(self) -> TriggerResponse:
        pass
