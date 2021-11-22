#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2020 rootadminWalker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
from copy import copy

import cv2 as cv
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes, ObjectBox

from core.Dtypes import BBox
from core.Nodes import Node
import message_filters


class PFResultShower(Node):
    IOU_THRESHOLD = 0.93

    def __init__(self):
        super(PFResultShower, self).__init__("pf_visualizer", anonymous=False)

        self.rgb_image = None
        self.estimated_box = None
        self.bridge = CvBridge()
        self.detection_boxes = []

        rospy.Subscriber(
            "/YD/boxes",
            ObjectBoxes,
            self.box_callback,
            queue_size=1
        )

        rospy.Subscriber(
            "/person_follower/estimated_target_box",
            ObjectBox,
            self.estimated_box_callback,
            queue_size=1
        )

        self.main()

    def box_callback(self, detections: ObjectBoxes):
        self.detection_boxes = BBox.from_ObjectBoxes(detections)
        self.detection_boxes = list(filter(lambda b: b.label.strip() == 'person', self.detection_boxes))
        self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(detections.source_img)

    def estimated_box_callback(self, estimated_box: ObjectBox):
        self.estimated_box = BBox.from_ObjectBox(estimated_box)

    @staticmethod
    def __draw_box_and_centroid(image, box, color, thickness, radius):
        box.draw(image, color, thickness)
        box.draw_centroid(image, color, radius)

    def main(self):
        last_box = None
        while not rospy.is_shutdown():
            srcframe = self.rgb_image
            if srcframe is None or not rospy.get_param('/person_follower/initialized'):
                continue

            frame = srcframe.copy()

            current_person_boxes = copy(self.detection_boxes)
            estimated_box = copy(self.estimated_box)
            current_state = rospy.get_param('/person_follower/state')

            for person_box in current_person_boxes:
                if person_box.iou_score_with(person_box) > PFResultShower.IOU_THRESHOLD and current_state == 'NORMAL':
                    continue
                self.__draw_box_and_centroid(frame, person_box, (32, 0, 255), 3, 3)

            if current_state == 'NORMAL':
                self.__draw_box_and_centroid(frame, estimated_box, (32, 255, 0), 5, 7)
            elif current_state == 'CONFIRM_REIDENTIFIED':
                self.__draw_box_and_centroid(frame, estimated_box, (255, 255, 0), 5, 7)
            elif current_state == 'CONFIRM_LOST':
                self.__draw_box_and_centroid(frame, last_box, (32, 255, 255), 5, 7)

            cv.imshow('frame', frame)
            key = cv.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

            if estimated_box is not None:
                last_box = copy(estimated_box)
            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = PFResultShower()
