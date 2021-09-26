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
import json

import dlib
import numpy as np
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes

from core.Nodes import Node


class FaceRecognition(Node):
    def __init__(self):
        super(FaceRecognition, self).__init__('face_desc_parser', anonymous=False)

        _shape_dat = rospy.get_param('~shape_dat')
        _desc_parser = rospy.get_param('~desc_parser')

        self.shape_predictor = dlib.shape_predictor(_shape_dat)
        self.face_desc_parser = dlib.face_recognition_model_v1(_desc_parser)

        self.bridge = CvBridge()
        self.faces = None

        self.desc_pub = rospy.Publisher(
            '~descriptors',
            ObjectBoxes,
            queue_size=1
        )

        self.landmarks_pub = rospy.Publisher(
            "~landmarks",
            ObjectBoxes,
            queue_size=1,
        )

        rospy.Subscriber(
            '/FD/faces',
            ObjectBoxes,
            self.face_callback,
            queue_size=1
        )

        self.main()

    def face_callback(self, faces: ObjectBoxes):
        self.faces = faces.boxes

    def main(self):
        while not rospy.is_shutdown():
            if self.faces is None:
                continue

            face_desc_msg = ObjectBoxes()

            faces = self.faces.copy()

            for face in faces:
                face.model = 'dlib_face_recognition'
                face_image = self.bridge.compressed_imgmsg_to_cv2(face.source_img)

                # Parsing part
                face_img_h, face_img_w, _ = face_image.shape
                full_image_rect = dlib.rectangle(0, 0, face_img_w, face_img_h)
                landmarks = self.shape_predictor(face_image, full_image_rect)
                face_descriptor = np.array(
                    self.face_desc_parser.compute_face_descriptor(face_image, landmarks)
                )

                face.label = json.dumps(list(face_descriptor))

            face_desc_msg.boxes = faces

            self.desc_pub.publish(face_desc_msg)

            self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = FaceRecognition()
