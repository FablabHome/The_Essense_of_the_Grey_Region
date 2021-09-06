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
from os import path

import color_transfer
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes, PFRobotData, PFWaypoints, ObjectBox
from home_robot_msgs.srv import PFInitializer, PFInitializerRequest, PFInitializerResponse
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage

from core.Detection import PersonReidentification
from core.Dtypes import BBox


class PersonFollower:
    H = 480
    W = 640
    CENTROID = (W // 2, H // 2)

    SIMIL_ERROR = 0.53
    STATE = 'NORMAL'  # SEARCHING, LOST

    # Timeouts
    LOST_TIMEOUT = rospy.Duration(1.5)

    # Waypoint maximize size and thickness
    WAYPOINT_MAX_RADIUS = 12
    WAYPOINT_THICKNESS = 2  # If thickness is -1 than the circle will be filled

    # Continuously collect descriptors settings
    DESCRIPTORS_COUNT = 20  # Descriptors maxlength
    COLLECT_TIMEOUT = 1  # Collecting timeout

    def __init__(self, person_extractor: PersonReidentification):
        self.person_extractor = person_extractor
        self.bridge = CvBridge()

        self.front_descriptor = self.back_descriptor = None
        self.target_box = self.last_box = self.tmp_box = None
        self.rgb_image = None
        self.front_img = self.back_img = None

        self.detection_boxes = []
        self.distance_and_boxes = {}
        self.waypoints = []
        self.descriptors = deque([], maxlen=PersonFollower.DESCRIPTORS_COUNT)
        # self.max_distance = 0

        self.initialized = False

        rospy.Service('pf_initialize', PFInitializer, self.initialized_cb)

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

    def box_callback(self, detections: ObjectBoxes):
        if self.initialized:
            self.detection_boxes = detections.boxes
            # max_distance = 0

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
    def __compare_descriptor(desc1, desc2):
        return np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))

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
        # confirm_timeout = rospy.get_rostime() + PersonFollower.CONFIRM_TIMEOUT
        waypoint_color = (32, 255, 0)

        similarities = []

        while not rospy.is_shutdown():
            srcframe = self.rgb_image
            if srcframe is None:
                continue

            self.distance_and_boxes = {}
            # Update self.last_box if the first target_box was confirmed
            if self.target_box is not None:
                self.last_box = self.target_box
            self.target_box = self.tmp_box = None

            max_similarity = 0.
            for det_box in self.detection_boxes:
                # Unpack the source image
                source_img = self.bridge.compressed_imgmsg_to_cv2(det_box.source_img)
                # Ignore detections which was not person
                if det_box.label.strip() != 'person':
                    continue

                person_box = BBox(det_box.x1, det_box.y1, det_box.x2, det_box.y2, label='person',
                                  score=det_box.score,
                                  source_img=source_img)

                # Draw the box in red as the base layer
                self.__draw_box_and_centroid(srcframe, person_box, (32, 0, 255), 3, 3)

                # Calculate distances between the last existance of the target
                if self.last_box is not None:
                    dist_between_target = self.last_box.calc_distance_between_point(person_box.centroid)
                    self.distance_and_boxes.update({dist_between_target: person_box})

                # # Matching styles with the original front and back image
                # matched_front = color_transfer.color_transfer(self.front_img, source_img)
                # matched_back = color_transfer.color_transfer(self.back_img, source_img)
                # matched_front = source_img
                # matched_back = source_img

                # Parse the current descriptor
                # matched_front_desc = self.person_extractor.parse_descriptor(matched_front, crop=False)
                # matched_back_desc = self.person_extractor.parse_descriptor(matched_back, crop=False)
                current_descriptor = self.person_extractor.parse_descriptor(source_img, crop=False)

                # Compare the similarity
                front_similarity = self.__compare_descriptor(current_descriptor, self.front_descriptor)
                back_similarity = self.__compare_descriptor(current_descriptor, self.back_descriptor)
                n = max(front_similarity, back_similarity)
                if front_similarity > n:
                    max_similarity = n
                # rospy.loginfo(f'{front_similarity},{back_similarity}')

                # if self.max_distance < distance_between_centroid < 250:
                if self.__similarity_lt(front_similarity) or self.__similarity_lt(back_similarity):
                    # cv.imshow('matched_front', matched_front)
                    # cv.imshow('matched_back', matched_back)
                    # cv.imshow('front_img', self.front_img)
                    PersonFollower.STATE = 'NORMAL'
                    self.target_box = person_box
                    self.__draw_box_and_centroid(srcframe, self.target_box, (32, 255, 0), 9, 9)
                    self.descriptors.append(current_descriptor)
                    similarities.append(n)

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
                        if len(self.distance_and_boxes) > 0:
                            # Use the box closest to the last existance of the target
                            self.tmp_box = self.distance_and_boxes[min(self.distance_and_boxes.keys())]
                            msg.follow_point = self.tmp_box.centroid
                            # Draw box and update waypoint_color
                            self.__draw_box_and_centroid(srcframe, self.tmp_box, (32, 255, 255), 5, 5)
                            waypoint_color = (32, 255, 255)
                    similarities.append(max_similarity)

                elif PersonFollower.STATE == 'LOST':
                    msg.follow_point = (-1, -1)
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
            else:
                self.box_publisher.publish(ObjectBox(
                    x1=self.last_box.x1,
                    y1=self.last_box.y1,
                    x2=self.last_box.x2,
                    y2=self.last_box.y2
                ))

            # Draw the waypoints when param 'toggle_waypoint_animation' is True
            if rospy.get_param('~toggle_waypoint_animation'):
                self.__draw_waypoints(srcframe, waypoint_color)

            # drown_image = node.bridge.cv2_to_compressed_imgmsg(rgb_image)
            # node.image_publisher.publish(drown_image)

            frame = srcframe
            cv.imshow('frame', frame)
            key = cv.waitKey(1)
            rospy.set_param('~state', PersonFollower.STATE)
            if key in [ord('q'), 27]:
                break

        cv.destroyAllWindows()
        plt.title('Stage 2 single person dark similarity test')
        axes = plt.gca()
        axes.set_ylim([0.0, 1.0])
        plt.plot(similarities)
        plt.savefig('with_color_transfer.png')
        plt.show()
        rospy.loginfo(np.mean(similarities))


if __name__ == '__main__':
    rospy.init_node('person_follower')
    base = RosPack().get_path('rcj_pcms_base') + '/..'
    bin_path = path.join(
        base, 'models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.bin')
    xml_path = path.join(
        base, 'models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml')
    net = cv.dnn.readNet(bin_path, xml_path)
    person_descriptor_extractor = PersonReidentification(net)
    node = PersonFollower(person_descriptor_extractor)
