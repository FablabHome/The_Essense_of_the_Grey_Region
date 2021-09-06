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

from abc import abstractmethod
from typing import Any, List
import numpy as np

import rospy


class ModelInput:
    @abstractmethod
    def preprocess_to(self, input_data) -> Any:
        return input_data

    @abstractmethod
    def rollback(self, blob) -> np.array:
        return blob


class Outputs:
    @abstractmethod
    def process_outputs(self, outputs: Any) -> Any:
        return outputs


class Detector:
    def __init__(
            self,
            detector: Any,
            image_processor: ModelInput,
            output_processor: Outputs,
            need_blob=False
    ):
        self.detector = detector
        self.image_processor = image_processor
        self.output_processor = output_processor
        self.need_blob = need_blob

        self.blob = None

    @abstractmethod
    def detect(self, image) -> Any:
        pass


class NodeProgram:
    def __init__(self, node_id):
        self.id = node_id

    @abstractmethod
    def serialize_output(self) -> Any:
        pass


class Node:
    def __init__(self, name: str = 'node', anonymous: bool = True):
        """

        Args:
            name: The name of the node, will be use in rospy.init_node
            anonymous: If you want to generate a random ID for the node

        """
        self.name = name
        self.anonymous = anonymous

        rospy.init_node(self.name, anonymous=self.anonymous)
        rospy.loginfo(f'Node {rospy.get_name()} Created')

    @abstractmethod
    def reset(self):
        """
        This method will reset every values in the Node
        """
        pass

    @staticmethod
    def spin():
        """
        Call rospy.spin to spin the node
        """
        rospy.spin()

    @staticmethod
    def wait_for_msg(topic, data_class):
        """
        You can use this method for waiting a msg with info coming out

        Args:
            topic: The topic you want to wait for
            data_class:

        Returns: None

        """
        rospy.loginfo(f'Waiting response from {topic}')
        rospy.wait_for_message(topic, data_class)
        rospy.loginfo(f'{topic}: Ok')
