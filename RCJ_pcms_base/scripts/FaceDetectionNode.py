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
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes, ObjectBox
from sensor_msgs.msg import CompressedImage


class FaceDetectionNode:
    def __init__(self, face_detector):
        self.face_detector = face_detector

        self.face_pub = rospy.Publisher(
            '~faces',
            ObjectBoxes,
            queue_size=1
        )

        rospy.Subscriber(
            rospy.get_param('~image_source'),
            CompressedImage,
            self.callback,
            queue_size=1
        )

        self.bridge = CvBridge()
        self.srcframe = None
        self.main()

    def callback(self, img: CompressedImage):
        self.srcframe = self.bridge.compressed_imgmsg_to_cv2(img)

    def main(self):
        while not rospy.is_shutdown():
            face_msg = ObjectBoxes()
            if self.srcframe is None:
                continue

            frame = self.srcframe.copy()
            H, W, _ = frame.shape

            blob = cv.dnn.blobFromImage(
                frame,
                size=(416, 416),
                scalefactor=1.0,
                mean=(0, 0, 0),
                swapRB=False,
                crop=False
            )
            self.face_detector.setInput(blob)
            faces = self.face_detector.forward()[0][0]
            for _, _, conf, x1, y1, x2, y2 in faces:
                if conf < 0.4:
                    continue

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                x2 = int(x2 * W)
                y2 = int(y2 * H)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                face_roi = frame[y1:y2, x1:x2].copy()

                face = ObjectBox()
                face.model = 'face-detection-0204'
                face.x1 = max(0, x1)
                face.y1 = max(0, y1)
                face.x2 = x2
                face.y2 = y2
                face.score = conf
                if face_roi is None:
                    continue
                face.source_img = self.bridge.cv2_to_compressed_imgmsg(face_roi)
                face_msg.boxes.append(face)

            face_msg.source_img = self.bridge.cv2_to_compressed_imgmsg(frame)
            self.face_pub.publish(face_msg)


if __name__ == '__main__':
    rospy.init_node('FD')
    bin_path = rospy.get_param('~bin_path')
    xml_path = rospy.get_param('~xml_path')
    face_detector = cv.dnn.readNet(bin_path, xml_path)
    node = FaceDetectionNode(face_detector)
