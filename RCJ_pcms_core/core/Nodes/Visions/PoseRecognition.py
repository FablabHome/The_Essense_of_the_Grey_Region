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

from ...Detection import PoseDetector, PoseRecognitionInput, PoseRecognitionProcess, PoseRecognitionDetector
from ...base_classes import NodeProgram
from ...tools import OpenPose
from ...Dtypes import BBox

from home_robot_msgs.msg import HumanPose

from tensorflow.keras.models import Model
import numpy as np


class PoseRecognition(NodeProgram):
    def __init__(
            self,
            node_id: str,
            pose_detector_entry: OpenPose,
            pose_recognition_entry: Model,
            input_shape: tuple
    ):
        super(PoseRecognition, self).__init__(node_id)

        self.input_shape = input_shape
        self.pose_detector_entry = pose_detector_entry
        self.pose_recognition_entry = pose_recognition_entry

        self.pose_detector = PoseDetector(self.pose_detector_entry)

        self.pose_image_processor = PoseRecognitionInput(
            padding=(50, 70),
            image_shape=self.input_shape
        )
        self.pose_output_processor = PoseRecognitionProcess()

        self.pose_recognizer = PoseRecognitionDetector(
            detector=self.pose_recognition_entry,
            image_processor=self.pose_image_processor,
            output_processor=self.pose_output_processor,
        )

        self.serialize_msg = HumanPose()

        self.pose_box = None

        self.pose_boxes = []

    def run(self, image: np.array) -> (np.array, np.array, BBox):
        pose_points = self.pose_detector.detect(image)
        for pose_point in pose_points:
            self.pose_detector_entry.draw(image, pose_point, thickness=5)

            self.pose_box = self.pose_recognizer.detect(pose_point)
            if self.pose_box is None:
                continue

            status = self.pose_box.label
            confidence = self.pose_box.score

            self.pose_box.draw(image)
            self.pose_box.putText_at_top(image, f'{status}: {confidence}')

            self.pose_boxes.append(self.pose_box)

        return image, pose_points, self.pose_boxes

    def serialize_output(self, outputs: np.array, **kwargs) -> HumanPose:
        person_count = outputs.shape[0]
        pose_boxes = kwargs.get('pose_boxes')

        outputs = outputs.reshape(
            person_count * 18 * 2
        )

        serialized_pose_boxes = list(map(
            lambda x: x.serialize_ros(),
            pose_boxes
        ))

        self.serialize_msg.points = outputs
        self.serialize_msg.person_count = person_count
        self.serialize_msg.pose_boxes = serialized_pose_boxes

        return self.serialize_msg
