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
import types
import warnings
from abc import ABC
from os import path

import rospy
from actionlib_msgs.msg import GoalID
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rospkg import RosPack
from std_msgs.msg import String

from ..tools import ManipulatorController, Speaker, Chassis, SLAMController


class MainProgram(ABC):
    """
    Abstract class for main program
    Available members:

    Services:
    self.speaker: Speaker Service to say things synchronously

    core.tools:
    self.manipulator: ManipulatorController
    self.chassis: Chassis
    self.speaker: Speaker
    self.slam: SLAMController

    Publishers:
    self.cancel_slam: Publisher to cancel the slam goal
    self.speaker_pub: Speaker to say things asynchronously
    self.goal_pub: Publishes a goal to the slam controller
    self.facial_pub: Change facial expressions of FacialDisplay.py

    Other:
    self.bridge: CvBridge()
    """
    def __init__(self):
        if not rospy.core.is_initialized():
            raise rospy.ROSException('Please initialize first')

        if 'codes' not in self.__dict__:
            raise AttributeError('No codes stated, please state it before using <super>')

        for code, action in self.codes.items():
            if type(action) is not types.MethodType:
                warnings.warn(
                    f"code <{code}>'s action '{action}' wasn't a method, this may cause errors",
                    ResourceWarning)

        self.speaker = Speaker()
        self.manipulator = ManipulatorController()
        self.chassis = Chassis()
        self.bridge = CvBridge()

        self.base = RosPack().get_path('rcj_pcms_base')
        config = path.join(self.base, 'config/points.json')
        self.slam = SLAMController(config=config)

        self.cancel_slam = rospy.Publisher(
            '/move_base/cancel',
            GoalID,
            queue_size=1
        )
        self.facial_pub = rospy.Publisher(
            '/home_edu/facial',
            String,
            queue_size=1
        )
        rospy.Subscriber(
            '~code',
            String,
            self.code_callback,
            queue_size=1
        )

    def code_callback(self, data):
        code = data.data
        self.codes[code]()
