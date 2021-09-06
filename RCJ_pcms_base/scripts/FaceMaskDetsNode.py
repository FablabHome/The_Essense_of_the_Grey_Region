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
import os

import cv2 as cv
import numpy as np
import rospy
import tensorflow as tf
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
cfgs = tf.compat.v1.ConfigProto(gpu_options=opts)
sess = tf.compat.v1.Session(config=cfgs)


class FaceMaskDetsNode:
    def __init__(self, mask_detector: tf.keras.Model):
        self.mask_detector = mask_detector
        self.mask_status_pub = rospy.Publisher(
            '~mask_is_on',
            ObjectBoxes,
            queue_size=1
        )
        rospy.Subscriber(
            '/FD/faces',
            ObjectBoxes,
            self.face_callback,
            queue_size=1
        )

        rospy.set_param('~kill', False)
        rospy.set_param('~lock', False)

        self.bridge = CvBridge()
        self.get_faces = ObjectBoxes()
        self.main()

    def face_callback(self, faces: ObjectBoxes):
        self.get_faces = faces

    def main(self):
        while not rospy.is_shutdown():
            if not rospy.get_param('~lock'):
                faces = self.get_faces
                for face in faces.boxes:
                    face_img = self.bridge.compressed_imgmsg_to_cv2(face.source_img)
                    blob = cv.cvtColor(face_img, cv.COLOR_BGR2RGB)
                    blob = cv.resize(blob, (224, 224))
                    blob = tf.keras.applications.mobilenet_v2.preprocess_input(blob)
                    blob = np.expand_dims(blob, axis=0)

                    (mask, withoutMask) = self.mask_detector.predict(blob)[0]
                    label = 1 if mask > withoutMask else 0
                    face.label = str(label)

                self.mask_status_pub.publish(faces)

            if rospy.get_param('~kill'):
                return


if __name__ == '__main__':
    rospy.init_node('FMD')
    model_path = rospy.get_param('~model_path')
    mask_detector = tf.keras.models.load_model(model_path)
    node = FaceMaskDetsNode(mask_detector)
