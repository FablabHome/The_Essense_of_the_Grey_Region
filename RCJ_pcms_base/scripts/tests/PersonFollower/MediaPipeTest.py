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

import mediapipe as mp
import rospy
from cv_bridge import CvBridge
import cv2 as cv
from sensor_msgs.msg import CompressedImage

from core.Nodes import Node


class TestMediaPipeFaceMesh(Node):
    def __init__(self):
        super(TestMediaPipeFaceMesh, self).__init__("test_mediapipe", anonymous=False)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_style = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

        self.bridge = CvBridge()
        self.srcframe = None

        rospy.Subscriber(
            "/image_raw/compressed",
            CompressedImage,
            self.callback,
            queue_size=1
        )

        self.main()

    def callback(self, image: CompressedImage):
        self.srcframe = self.bridge.compressed_imgmsg_to_cv2(image)

    def main(self):
        with self.mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as face_mesh:
            while not rospy.is_shutdown():
                if self.srcframe is None:
                    continue

                frame = self.srcframe.copy()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame.flags.writable = False
                results = face_mesh.process(frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_style.get_default_face_mesh_tesselation_style(),
                        )
                        self.mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_style.get_default_face_mesh_contours_style(),
                        )

                cv.imshow('frame', frame)
                key = cv.waitKey(1) & 0xFF
                if key in [27, ord('q')]:
                    break

                self.rate.sleep()

    def reset(self):
        pass


if __name__ == '__main__':
    node = TestMediaPipeFaceMesh()
