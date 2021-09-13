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

import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes

from core.Nodes import Node


class GenderDetectionNode(Node):
    def __init__(self, age_gender: cv.dnn_Net):
        self.ag_recognizer = age_gender

        self.result_pub = rospy.Publisher(
            '~details',
            ObjectBoxes,
            queue_size=1
        )

        rospy.Subscriber(
            '/FD/faces',
            ObjectBoxes,
            self.callback,
            queue_size=1
        )

        self.bridge = CvBridge()
        self.detected_faces = None
        self.main()

    def callback(self, faces: ObjectBoxes):
        self.detected_faces = faces

    def main(self):
        while not rospy.is_shutdown():
            faces = self.detected_faces
            if faces is None:
                continue

            frame = self.bridge.compressed_imgmsg_to_cv2(faces.source_img)
            for face in faces.boxes:
                face_roi = self.bridge.compressed_imgmsg_to_cv2(face.source_img)
                blob = cv.dnn.blobFromImage(
                    face_roi,
                    size=(62, 62),
                    scalefactor=1.0,
                    mean=(0, 0, 0),
                    swapRB=False,
                    crop=False
                )
                self.ag_recognizer.setInput(blob)
                p_age, p_gender = self.ag_recognizer.forward(['age_conv3', 'prob'])
                gender = np.argmax(p_gender)
                age = int(p_age[0][0][0][0] * 100)

                face.label = f'{gender}:{age}'

                # gender_txt = 'Male' if gender == 1 else 'Female'
                # color = (255, 100, 32) if gender == 1 else (32, 0, 255)
                #
                # age_txt = f'Looks like {age}'
                # cv.putText(frame, gender_txt, (face.x1, face.y1 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                # cv.putText(frame, age_txt, (face.x1, face.y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                # cv.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)

            # cv.imshow('frame', frame)
            # key = cv.waitKey(1) & 0xFF
            # if key in [27, ord('q')]:
            #     break
            self.result_pub.publish(faces)

    def reset(self):
        pass


if __name__ == '__main__':
    rospy.init_node('GAD')
    bin_path = rospy.get_param('~bin_path')
    xml_path = rospy.get_param('~xml_path')
    gender_age = cv.dnn.readNet(bin_path, xml_path)
    node = GenderDetectionNode(gender_age)
