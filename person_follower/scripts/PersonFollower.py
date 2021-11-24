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

from copy import deepcopy
from typing import List

import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes, ObjectBox
from home_robot_msgs.srv import PFInitializer, PFInitializerRequest, PFInitializerResponse, ResetPF, ResetPFRequest
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger

from core.Detection import PersonReidentification
from core.Dtypes import BBox
from core.Nodes import Node


class PersonFollower(Node):
    # TODO: Separate the showing fragment to another node like ShowFaceResult
    H = 480
    W = 640
    CENTROID = (W // 2, H // 2)

    SIMIL_ERROR = 0.55
    STATE = 'NORMAL'  # CONFIRM_LOST, LOST, CONFIRM_REIDENTIFIED

    # Timeouts
    CONFIRM_LOST_TIMEOUT = rospy.Duration(1.5)
    CONFIRM_REIDENTIFIED_TIMEOUT = rospy.Duration(0.5)

    # Threshold for the system to determine if a box was a close box
    CLOSE_BOX_IOU_THRESHOLD = 0.83

    SEARCH_BOX_PADDING = (25, 25)

    def __init__(self):
        super(PersonFollower, self).__init__('person_follower', anonymous=False)

        bin_path = rospy.get_param('~bin_path')
        xml_path = rospy.get_param('~xml_path')
        self.person_reid = PersonReidentification(bin_path, xml_path)

        self.bridge = CvBridge()

        self.front_descriptor = self.back_descriptor = None
        self.target_box = self.last_detected_box = None
        self.rgb_image = None
        self.front_img = self.back_img = None

        self.detection_boxes = []

        # Initialization host building
        self.initialized = False
        rospy.Service('pf_initialize', PFInitializer, self.initialized_cb)

        # The reset service server
        rospy.Service('pf_reset', ResetPF, self.on_reset)
        self.__question_answered = False  # Just for playing

        # Proxy of PFInitialize reset service
        self.reset_initializer = rospy.ServiceProxy('pf_init_reset', Trigger)

        # This publisher will only publish the box which the system was currently following
        self.current_following_box_pub = rospy.Publisher(
            '~current_following_box',
            ObjectBox,
            queue_size=1
        )

        # This publisher will publish the box whenever person re-id recognized the box
        self.estimated_target_publisher = rospy.Publisher(
            "~estimated_target_box",
            ObjectBox,
            queue_size=1,
        )

        self.image_publisher = rospy.Publisher(
            '/PF/drown_image',
            CompressedImage,
            queue_size=1
        )

        rospy.Subscriber(
            '/YD/boxes',
            ObjectBoxes,
            self.box_callback,
            queue_size=1
        )

        rospy.set_param('~initialized', False)
        rospy.set_param("~state", '')

        self.main()

    def initialized_cb(self, req: PFInitializerRequest):
        self.front_descriptor = np.array(req.front_descriptor)
        self.back_descriptor = np.array(req.back_descriptor)

        self.front_img = self.bridge.compressed_imgmsg_to_cv2(req.front_img)
        self.back_img = self.bridge.compressed_imgmsg_to_cv2(req.back_img)

        self.initialized = True
        rospy.set_param('~initialized', self.initialized)
        return PFInitializerResponse(True)

    def on_reset(self, req: ResetPFRequest):
        # Just for playing
        rospy.set_param("~todays_question", "V2hvJ3MgdGhlIGFic29sdXRlIGdvZCBvZiBoeXBlcmRlYXRoPw==")
        answer = 'QXNyaWVsIERyZWVtdXJy'

        user_answer = req.answer

        if user_answer == answer:
            self.reset()
            self.reset_initializer()

            return True
        else:
            return False

    def box_callback(self, detections: ObjectBoxes):
        if self.initialized:
            self.detection_boxes = BBox.from_ObjectBoxes(detections)
            self.detection_boxes = list(filter(lambda b: b.label.strip() == 'person', self.detection_boxes))

            self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(detections.source_img)
            H, W, _ = self.rgb_image.shape

            PersonFollower.W = W
            PersonFollower.CENTROID = (PersonFollower.W // 2, PersonFollower.H // 2)

    @staticmethod
    def __similarity_lt(similarity):
        return similarity > PersonFollower.SIMIL_ERROR

    def find_target_person(self, person_boxes: List[BBox]):
        for person_box in person_boxes:
            current_descriptor = self.person_reid.extract_descriptor(person_box.source_img, crop=False)

            # Compare the descriptors
            front_similarity = self.person_reid.compare_descriptors(current_descriptor, self.front_descriptor)
            back_similarity = self.person_reid.compare_descriptors(current_descriptor, self.back_descriptor)

            if self.__similarity_lt(front_similarity) or self.__similarity_lt(back_similarity):
                return person_box
        else:
            return None

    @staticmethod
    def find_tmp_person(search_box: BBox, person_boxes: List[BBox]):
        for person_box in person_boxes:
            if person_box.is_inBox(search_box):
                return person_box
        else:
            return None

    @staticmethod
    def find_overlapped_boxes(target_calc_box: BBox, person_boxes: List[BBox]):
        return list(filter(
            lambda b: b.iou_score_with(target_calc_box) >= PersonFollower.CLOSE_BOX_IOU_THRESHOLD,
            person_boxes
        ))

    def main(self):
        # Initialize the timeouts
        confirm_lost_timeout = rospy.get_rostime() + PersonFollower.CONFIRM_LOST_TIMEOUT
        confirm_reidentified_timeout = rospy.get_rostime() + PersonFollower.CONFIRM_REIDENTIFIED_TIMEOUT

        while not rospy.is_shutdown():
            if not self.initialized:
                continue

            if self.rgb_image is None:
                continue

            # Update self.last_box only if the target_box was confirmed
            if self.target_box is not None:
                self.last_detected_box = deepcopy(self.target_box)
            self.target_box = None

            # Copy the detection boxes out for safety
            current_detection_boxes = deepcopy(self.detection_boxes)

            # Get the target boxes
            self.target_box = self.find_target_person(current_detection_boxes)
            current_following_box = BBox(label='unrecognized')

            if self.target_box:
                # If the current state is NORMAL, publish the box out
                if PersonFollower.STATE == 'NORMAL':
                    current_following_box = self.target_box

                # If the current state is CONFIRM_LOST, then just jump
                # back to NORMAL
                elif PersonFollower.STATE == 'CONFIRM_LOST':
                    PersonFollower.STATE = 'NORMAL'

                # If the current state is LOST, then go into CONFIRM_REIDENTIFIED
                elif PersonFollower.STATE == 'LOST':
                    confirm_reidentified_timeout = rospy.get_rostime() + PersonFollower.CONFIRM_REIDENTIFIED_TIMEOUT
                    PersonFollower.STATE = 'CONFIRM_REIDENTIFIED'

                # If current state is CONFIRM_REIDENTIFIED, wait for the timeout
                elif PersonFollower.STATE == 'CONFIRM_REIDENTIFIED':
                    if rospy.get_rostime() - confirm_reidentified_timeout >= rospy.Duration(0):
                        PersonFollower.STATE = 'NORMAL'

                # Publish the estimated box without dealing with states
                self.estimated_target_publisher.publish(self.target_box.serialize_as_ObjectBox())
            else:
                # If the program lost the target, get in CONFIRM_LOST to confirm
                # if the target was truly lost
                if PersonFollower.STATE == 'NORMAL':
                    PersonFollower.STATE = 'CONFIRM_LOST'
                    confirm_lost_timeout = rospy.get_rostime() + PersonFollower.CONFIRM_LOST_TIMEOUT

                # If the program was confirming lost, then wait for the timeout
                # follow a person which was in the padding_box of our person's last existent
                elif PersonFollower.STATE == 'CONFIRM_LOST':
                    # If the timeout has exceeded, then the program will consider the target
                    # was truly lost
                    if rospy.get_rostime() - confirm_lost_timeout >= rospy.Duration(0):
                        PersonFollower.STATE = 'LOST'

                    # Follow a temporarily person which is inside the
                    # padding_box of the last_detected_box
                    search_box = self.last_detected_box.generate_padding_box(
                        padding=PersonFollower.SEARCH_BOX_PADDING,
                        shape=(PersonFollower.H, PersonFollower.W)
                    )

                    tmp_box = self.find_tmp_person(search_box, current_detection_boxes)
                    if tmp_box:
                        current_following_box = tmp_box

                # If the state was CONFIRM_REIDENTIFIED, it will just went back into LOST
                elif PersonFollower.STATE == 'CONFIRM_REIDENTIFIED':
                    PersonFollower.STATE = 'LOST'

            # Set the current state
            rospy.set_param("~state", PersonFollower.STATE)

            # Publish the to-follow box to the PFRobotHandler
            self.current_following_box_pub.publish(current_following_box.serialize_as_ObjectBox())

            self.rate.sleep()

    def reset(self):
        self.detection_boxes = []
        self.initialized = False
        # self.front_descriptor = self.back_descriptor = None
        cv.destroyAllWindows()


if __name__ == '__main__':
    node = PersonFollower()
