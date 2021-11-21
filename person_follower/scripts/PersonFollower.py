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

from collections import deque
from copy import copy

import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes, PFRobotData, PFWaypoints, ObjectBox
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
    STATE = 'NORMAL'  # SEARCHING, LOST

    # Timeouts
    LOST_TIMEOUT = rospy.Duration(1.5)

    # Waypoint maximize size and thickness
    WAYPOINT_MAX_RADIUS = 12
    WAYPOINT_THICKNESS = 2  # If thickness is -1 than the circle will be filled

    # The duration of person reid if the state was LOST
    PERSON_REID_DURATION = 0.5

    def __init__(self):
        super(PersonFollower, self).__init__('person_follower', anonymous=False)

        bin_path = rospy.get_param('~bin_path')
        xml_path = rospy.get_param('~xml_path')
        self.person_reid = PersonReidentification(bin_path, xml_path)

        self.bridge = CvBridge()

        self.front_descriptor = self.back_descriptor = None
        self.target_box = self.last_box = self.tmp_box = None
        self.rgb_image = None
        self.front_img = self.back_img = None

        self.detection_boxes = []
        self.distance_and_boxes = {}
        self.waypoints = []

        # Record the statuses for re-identification
        q_len = PersonFollower.PERSON_REID_DURATION / (1 / Node.ROS_RATE)
        self.status_queue = deque([True], maxlen=int(q_len))
        # This variable is to monitor if the target is recognized in that loop
        self.recognized = False

        # Initialization host building
        self.initialized = False
        rospy.Service('pf_initialize', PFInitializer, self.initialized_cb)

        # The reset service server
        rospy.Service('pf_reset', ResetPF, self.on_reset)
        self.__question_answered = False  # Just for playing

        # Proxy of PFInitialize reset service
        self.reset_initializer = rospy.ServiceProxy('pf_init_reset', Trigger)

        self.robot_handler_publisher = rospy.Publisher(
            '/PFRHandler/pf_data',
            PFRobotData,
            queue_size=1
        )

        self.image_publisher = rospy.Publisher(
            '/PF/drown_image',
            CompressedImage,
            queue_size=1
        )

        self.box_publisher = rospy.Publisher(
            '~target_box',
            ObjectBox,
            queue_size=1
        )

        rospy.Subscriber(
            '/YD/boxes',
            ObjectBoxes,
            self.box_callback,
            queue_size=1
        )

        rospy.Subscriber(
            '/record_waypoint/waypoints',
            PFWaypoints,
            self.waypoints_callback,
            queue_size=1
        )

        rospy.set_param('~lost_target', False)
        rospy.set_param('~toggle_waypoint_animation', True)
        rospy.set_param('~waypoint_max_radius', PersonFollower.WAYPOINT_MAX_RADIUS)
        rospy.set_param('~waypoint_thickness', PersonFollower.WAYPOINT_THICKNESS)

        self.main()

    def initialized_cb(self, req: PFInitializerRequest):
        self.front_descriptor = np.array(req.front_descriptor)
        self.back_descriptor = np.array(req.back_descriptor)

        self.front_img = self.bridge.compressed_imgmsg_to_cv2(req.front_img)
        self.back_img = self.bridge.compressed_imgmsg_to_cv2(req.back_img)

        self.initialized = True
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

    def waypoints_callback(self, waypoints: PFWaypoints):
        serialized_waypoints = waypoints.waypoints
        self.waypoints = list(map(lambda w: w.waypoint, serialized_waypoints))

    @staticmethod
    def __similarity_lt(similarity):
        return similarity > PersonFollower.SIMIL_ERROR

    @staticmethod
    def __draw_box_and_centroid(image, box, color, thickness, radius):
        box.draw(image, color, thickness)
        box.draw_centroid(image, color, radius)

    def __draw_waypoints(self, image, color):
        if len(self.waypoints) == 0:
            return

        radius_increase = rospy.get_param('~waypoint_max_radius') / len(self.waypoints)
        current_radius = 0
        for waypoint in self.waypoints:
            current_radius += radius_increase
            cv.circle(image, waypoint, int(current_radius), color, rospy.get_param('~waypoint_thickness'))

    def main(self):
        # Initialize the timeouts
        lost_timeout = rospy.get_rostime() + PersonFollower.LOST_TIMEOUT

        # Initialize the waypoint color
        waypoint_color = (32, 255, 0)

        while not rospy.is_shutdown():
            if not self.initialized:
                continue

            if self.rgb_image is None:
                continue

            srcframe = self.rgb_image.copy()

            self.distance_and_boxes = {}
            # Update self.last_box only if the target_box was confirmed
            if self.target_box is not None:
                self.last_box = self.target_box
            self.target_box = self.tmp_box = None

            self.recognized = False

            # Copy the detection boxes out for safety
            current_detection_boxes = copy(self.detection_boxes)

            for person_box in current_detection_boxes:
                # Ignore detections which was not person
                if person_box.label != 'person':
                    continue

                # Draw the box in red as the base layer
                self.__draw_box_and_centroid(srcframe, person_box, (32, 0, 255), 3, 3)

                # Calculate distances between the last existance of the target
                if self.last_box is not None:
                    dist_between_target = self.last_box.calc_distance_between_point(person_box.centroid)
                    self.distance_and_boxes.update({dist_between_target: person_box})

                current_descriptor = self.person_reid.extract_descriptor(person_box.source_img, crop=False)

                # Compare the similarity
                front_similarity = self.person_reid.compare_descriptors(current_descriptor, self.front_descriptor)
                back_similarity = self.person_reid.compare_descriptors(current_descriptor, self.back_descriptor)
                # rospy.loginfo(f'{front_similarity},{back_similarity}')

                if (self.__similarity_lt(front_similarity) or self.__similarity_lt(
                        back_similarity)) and not self.recognized:
                    # Append the status queue
                    self.status_queue.append(True)
                    self.recognized = True

                    # Confirmation with status queue if the state was LOST
                    if PersonFollower.STATE == 'LOST':
                        if not all(self.status_queue):
                            self.__draw_box_and_centroid(srcframe, person_box, (255, 255, 0), 5, 5)
                            continue
                        PersonFollower.STATE = 'NORMAL'
                    elif PersonFollower.STATE == 'SEARCHING':
                        PersonFollower.STATE = 'NORMAL'

                    self.target_box = copy(person_box)
                    self.__draw_box_and_centroid(srcframe, self.target_box, (32, 255, 0), 9, 9)

            msg = PFRobotData()
            msg.follow_point = (-1, -1)

            # Publishing data to the robot handler
            if self.target_box is None:
                if PersonFollower.STATE == 'NORMAL':
                    PersonFollower.STATE = 'SEARCHING'
                    lost_timeout = rospy.get_rostime() + PersonFollower.LOST_TIMEOUT
                elif PersonFollower.STATE == 'SEARCHING':
                    if rospy.get_rostime() - lost_timeout >= rospy.Duration(0):
                        PersonFollower.STATE = 'LOST'
                    else:
                        # Follow the last exist box's centroid
                        self.tmp_box = self.last_box
                        msg.follow_point = self.tmp_box.centroid
                        self.__draw_box_and_centroid(srcframe, self.tmp_box, (32, 255, 255), 5, 5)
                        waypoint_color = (32, 255, 255)

                elif PersonFollower.STATE == 'LOST':
                    msg.follow_point = (-1, -1)
                    if not self.recognized:
                        self.status_queue.append(False)
            else:
                msg.follow_point = self.target_box.centroid
                waypoint_color = (32, 255, 0)

            # Publish the data to the robot handler
            self.robot_handler_publisher.publish(msg)
            if self.target_box is not None:
                self.box_publisher.publish(ObjectBox(
                    x1=self.target_box.x1,
                    y1=self.target_box.y1,
                    x2=self.target_box.x2,
                    y2=self.target_box.y2
                ))

            # Draw the waypoints when param 'toggle_waypoint_animation' is True
            if rospy.get_param('~toggle_waypoint_animation'):
                self.__draw_waypoints(srcframe, waypoint_color)

            # drown_image = node.bridge.cv2_to_compressed_imgmsg(rgb_image)
            # node.image_publisher.publish(drown_image)

            rospy.set_param('~state', PersonFollower.STATE)

            frame = srcframe.copy()
            cv.imshow('frame', frame)
            key = cv.waitKey(1)
            if key in [ord('q'), 27]:
                break

            self.rate.sleep()

        cv.destroyAllWindows()

    def reset(self):
        self.detection_boxes = []
        self.initialized = False
        # self.front_descriptor = self.back_descriptor = None
        cv.destroyAllWindows()


if __name__ == '__main__':
    node = PersonFollower()
