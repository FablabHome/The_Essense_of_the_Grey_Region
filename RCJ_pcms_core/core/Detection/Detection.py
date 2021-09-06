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

from ..base_classes import *
from ..Dtypes import BBox, posToBBox, PoseGesture
from ..tools import OpenPose

from PIL import Image
from keras_yolo3_qqwweee.yolo import YOLO
from keras.models import Model
import tensorflow as tf
import numpy as np
import cv2 as cv
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
cfgs = tf.compat.v1.ConfigProto(gpu_options=opts)
sess = tf.compat.v1.Session(config=cfgs)


class YOLOInput(ModelInput):
    def preprocess_to(self, input_data: np.array) -> Image:
        return Image.fromarray(input_data)

    def rollback(self, blob: Image) -> np.array:
        return np.array(blob)


class YOLOProcess(Outputs):
    def process_outputs(self, outputs: list) -> dict:
        box_count = outputs[0]
        out_boxes = outputs[1]
        predicted_classes = outputs[2]

        outputs = {
            'box_count': box_count,
            'out_boxes': posToBBox(out_boxes=out_boxes, labels=predicted_classes),
        }
        return outputs


class PoseRecognitionInput(ModelInput):
    def __init__(self, padding=(50, 50), image_shape=(640, 480)):
        self.padding = padding
        self.image_shape = image_shape

    def preprocess_to(self, input_data: np.array) -> (BBox, np.array):
        one_person_points = PoseGesture(pose_points=input_data)

        black_board = one_person_points.to_black_board()
        pose_box = one_person_points.to_box()

        pose_box = posToBBox([pose_box])[0]

        return pose_box, black_board

    def rollback(self, blob) -> np.array:
        pass


class PersonReIdentifyInput(ModelInput):
    def preprocess_to(self, input_data) -> np.array:
        return cv.dnn.blobFromImage(input_data, size=(128, 256))

    def rollback(self, blob) -> np.array:
        pass


class PoseRecognitionProcess(Outputs):
    def __init__(self):
        self.labels = ['fall', 'sit', 'squat', 'stand']

    def process_outputs(self, outputs: np.array) -> (str, float):
        idx = np.argmax(outputs)

        label = self.labels[idx]

        confidence = outputs[0][idx]
        confidence = round(confidence, 2)

        return label, confidence


class YOLODetector(Detector):
    def _input_process(self, input_data) -> Any:
        pass

    def _output_process(self) -> Any:
        pass

    def __init__(
            self,
            detector: YOLO,
            image_processor: ModelInput,
            output_processor: Outputs,
            need_blob=False
    ):
        super(YOLODetector, self).__init__(detector, image_processor, output_processor, need_blob)

    def detect(self, image) -> dict:
        self.blob = self.image_processor.preprocess_to(image)

        _, box_count, out_boxes, predicted_classes = self.detector.detect_image(self.blob)

        result = self.output_processor.process_outputs(
            [box_count, out_boxes, predicted_classes]
        )

        if self.need_blob:
            result.update({'blob': self.blob})

        return result


class PoseDetector(Detector):
    def __init__(
            self,
            detector: OpenPose,
            image_processor=None,
            output_processor=None,
            need_blob=False
    ):
        super(PoseDetector, self).__init__(detector, image_processor, output_processor, need_blob)

    def detect(self, image) -> np.array:
        points = self.detector.detect(image)
        return points


class PoseRecognitionDetector(Detector):
    def __init__(
            self,
            detector: Model,
            image_processor: ModelInput,
            output_processor: Outputs,
            need_blob=False
    ):
        super(PoseRecognitionDetector, self).__init__(detector, image_processor, output_processor, need_blob)

    def detect(self, image: np.array):
        result = self.image_processor.preprocess_to(image)
        if result is not None:
            pose_box, self.blob = result
        else:
            return

        predicts = self.detector.predict(np.array([self.blob / 255.0]))

        status, confidence = self.output_processor.process_outputs(predicts)

        pose_box.label = status
        pose_box.score = confidence

        return pose_box
