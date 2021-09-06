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
from typing import Any

import dlib
import genpy

from .vision_abstract import VisionNodeProgram
from ...Dtypes.FaceProcess import FaceUser, FaceUserManager


class FaceRecognition(VisionNodeProgram):
    def __init__(
            self,
            node_id,
            landmark_vector_convertor: dlib.face_recognition_model_v1,
            landmark_predictor: dlib.shape_predictor
    ):
        super(FaceRecognition, self).__init__(node_id)
        self.landmark_vector_convertor = landmark_vector_convertor
        self.landmark_predictor = landmark_predictor

    def run(self, serialize=False, **input_data) -> Any:
        image = input_data.get('image')
        box: dlib.rectangle = input_data.get('box')
        user_manager: FaceUserManager = input_data.get('user_manager')

        landmarks = self.landmark_predictor(image, box)
        descriptor = self.landmark_vector_convertor.compute_face_descriptor(image, landmarks)

        temp_user = FaceUser('', description=descriptor)

        recognized_user = user_manager.sign_in(temp_user)
        self.vision_output['found_user'] = recognized_user
        return recognized_user

    def serialize_output(self) -> genpy.Message:
        return self.vision_output
