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

from ...Detection import YOLOInput, YOLOProcess, YOLODetector
from .vision_abstract import VisionNodeProgram

from home_robot_msgs.msg import ObjectBox, ObjectBoxes

from keras_yolo3_qqwweee.yolo import YOLO
from os import path
import numpy as np


class YOLODetection(VisionNodeProgram):
    def __init__(
            self,
            node_id: str,
    ):
        super(YOLODetection, self).__init__(node_id)

        self.detector_entry = YOLO(
            model_path=path.realpath('../models/YOLO/yolov3_416.h5'),
            anchors_path=path.realpath('../model_data/YOLO/yolo_anchors.txt'),
            classes_path=path.realpath('../model_data/YOLO/coco_classes.txt'),
            font_path=path.realpath('../font/FiraMono-Medium.otf')
        )

        self.image_processor = YOLOInput()
        self.outputs_processor = YOLOProcess()

        self.detector = YOLODetector(
            image_processor=self.image_processor,
            output_processor=self.outputs_processor,
            detector=self.detector_entry
        )

        self.outputs = {}

        self.serialize_box = ObjectBox()
        self.output_msg = ObjectBoxes
        self.serialize_boxes = self.output_msg()

    def run(self, input_data: np.array, serialize=True) -> dict:
        self.outputs = result = self.detector.detect(input_data)
        if serialize:
            result = self.serialize_output()

        return result

    def serialize_output(self) -> ObjectBoxes:
        self.serialize_boxes = ObjectBoxes()
        for box in self.outputs['out_boxes']:
            self.serialize_box.label = box.label
            self.serialize_box.model = 'yolo'

            self.serialize_box.x1 = box.x1
            self.serialize_box.y1 = box.y1
            self.serialize_box.x2 = box.x2
            self.serialize_box.y2 = box.y2

            self.serialize_boxes.boxes.append(self.serialize_box)

        return self.serialize_boxes
