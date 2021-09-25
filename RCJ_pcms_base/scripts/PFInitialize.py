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

from os import path

import cv2 as cv
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes, ObjectBox
from home_robot_msgs.srv import PFInitializer, PFInitializerRequest
from mr_voice.srv import SpeakerSrv
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from core.Detection import PersonReidentification
from core.Dtypes import BBox
from core.Nodes import Node


class PFInitialize(Node):
    H = 480
    W = 640
    INIT_TIMEOUT = 1.8
    INIT_BOX_SIZE = 0.7

    def __init__(self):
        super(PFInitialize, self).__init__('pf_initializer')

        base = RosPack().get_path('rcj_pcms_base') + '/..'
        bin_path = path.join(
            base, 'models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.bin')
        xml_path = path.join(
            base, 'models/intel/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml')
        net = cv.dnn.readNet(bin_path, xml_path)
        self.person_desc_extractor = PersonReidentification(net)

        self.rgb_image = None
        rospy.wait_for_service('pf_initialize')
        self.speaker_pub = rospy.Publisher(
            '/speaker/text',
            String,
            queue_size=1
        )

        self.image_publisher = rospy.Publisher(
            '/PF/drown_image',
            CompressedImage,
            queue_size=1
        )

        self.init_box_publisher = rospy.Publisher(
            '~init_box',
            ObjectBox,
            queue_size=1
        )

        self.yolo_sub = rospy.Subscriber(
            '/YD/boxes',
            ObjectBoxes,
            self.yolo_boxes_callback,
            queue_size=1
        )
        self.speaker_srv = rospy.ServiceProxy('/speaker/text', SpeakerSrv)
        self.call_person_follower = rospy.ServiceProxy('pf_initialize', PFInitializer)
        rospy.set_param('~initialized', False)

        self.bridge = CvBridge()
        self.init_box = self.__generate_initial_box()
        self.inside_init_box = []
        self.main()

    def yolo_boxes_callback(self, detections: ObjectBoxes):
        inside_init_box = []
        detection_boxes = detections.boxes
        self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(detections.source_img)
        PFInitialize.H, PFInitialize.W, _ = self.rgb_image.shape
        for det_box in detection_boxes:
            source_img = self.bridge.compressed_imgmsg_to_cv2(det_box.source_img)
            person_box = BBox(
                det_box.x1, det_box.y1, det_box.x2, det_box.y2, label=det_box.label.strip(), score=det_box.score,
                source_img=source_img
            )
            if person_box.label != 'person' or person_box.area < 51120:
                continue

            # rospy.loginfo(person_box.is_inBox(self.init_box))
            if person_box.is_inBox(self.init_box):
                inside_init_box.append(person_box)
        self.inside_init_box = inside_init_box

    @classmethod
    def __generate_initial_box(cls):
        ch, cw = cls.H // 2, cls.W // 2
        return BBox(cw, ch, cw, ch, padding=(cw * PFInitialize.INIT_BOX_SIZE, cls.H),
                    shape=(cls.H, cls.W)).padding_box

    def reset(self):
        pass

    def main(self):
        self.speaker_srv("Hi, I'm a follower, please let me see your front and wait for 3 seconds")

        init_timeout = rospy.Duration(PFInitialize.INIT_TIMEOUT)
        timeout = rospy.get_rostime() + init_timeout

        info_text = ''
        inside_init_box = []

        while rospy.get_rostime() - timeout <= init_timeout:
            inside_init_box = self.inside_init_box
            if len(inside_init_box) != 1:
                timeout = rospy.get_rostime() + init_timeout
                if len(inside_init_box) == 0:
                    info_text = 'Please stand inside the blue box'
                elif len(inside_init_box) > 1:
                    info_text = 'Only 1 person can stand inside the blue box'
            else:
                info_text = 'Please wait for a few seconds'

            rgb_image = self.rgb_image
            if rgb_image is None:
                continue

            self.init_box.draw(rgb_image, (255, 0, 32), 3)
            cv.putText(rgb_image, info_text, (10, PFInitialize.H - 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # drown_image = self.bridge.cv2_to_compressed_imgmsg(rgb_image)
            # self.image_publisher.publish(drown_image)
            cv.imshow('frame', rgb_image)
            cv.waitKey(1)

        self.init_box_publisher.publish(ObjectBox(
            x1=self.init_box.x1,
            y1=self.init_box.y1,
            x2=self.init_box.x2,
            y2=self.init_box.y2
        ))
        rospy.loginfo('ok')
        self.speaker_srv('Please let me see your back in 3 seconds')
        front_image = inside_init_box[0].source_img
        serialized_front_img = self.bridge.cv2_to_compressed_imgmsg(front_image)
        front_descriptor = self.person_desc_extractor.parse_descriptor(front_image)
        front_descriptor = front_descriptor[0].tolist()

        timeout = rospy.get_rostime() + init_timeout
        while rospy.get_rostime() - timeout <= init_timeout:
            inside_init_box = self.inside_init_box
            if len(inside_init_box) != 1:
                timeout = rospy.get_rostime() + init_timeout
                if len(inside_init_box) == 0:
                    info_text = 'Please stand inside the blue box'
                elif len(inside_init_box) > 1:
                    info_text = 'Only 1 person can stand inside the blue box'
            else:
                info_text = 'Please turn back'

            rgb_image = self.rgb_image
            self.init_box.draw(rgb_image, (255, 0, 32), 3)

            cv.putText(rgb_image, info_text, (10, PFInitialize.H - 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # drown_image = self.bridge.cv2_to_compressed_imgmsg(rgb_image)
            # self.image_publisher.publish(drown_image)
            cv.imshow('frame', rgb_image)
            cv.waitKey(1)

        back_image = inside_init_box[0].source_img
        serialized_back_img = self.bridge.cv2_to_compressed_imgmsg(back_image)
        back_descriptor = self.person_desc_extractor.parse_descriptor(back_image)
        back_descriptor = back_descriptor[0].tolist()

        request_srv = PFInitializerRequest()
        request_srv.front_img = serialized_front_img
        request_srv.back_img = serialized_back_img
        request_srv.front_descriptor = front_descriptor
        request_srv.back_descriptor = back_descriptor

        self.call_person_follower(request_srv)
        rospy.set_param('~initialized', True)
        self.speaker_srv("I've remembered you, you can start walking")


if __name__ == '__main__':
    node = PFInitialize()

    # node.speaker_srv("Hi, I'm a follower, please let me see your front and wait for 3 seconds")
    #
    # init_timeout = rospy.Duration(node.INIT_TIMEOUT)
    # timeout = rospy.get_rostime() + init_timeout
    #
    # info_text = ''
    # inside_init_box = []
    #
    # while rospy.get_rostime() - timeout <= init_timeout:
    #     inside_init_box = node.inside_init_box
    #     if len(inside_init_box) != 1:
    #         timeout = rospy.get_rostime() + init_timeout
    #         if len(inside_init_box) == 0:
    #             info_text = 'Please stand inside the blue box'
    #         elif len(inside_init_box) > 1:
    #             info_text = 'Only 1 person can stand inside the blue box'
    #     else:
    #         info_text = 'Please wait for a few seconds'
    #
    #     rgb_image = node.rgb_image
    #     if rgb_image is None:
    #         continue
    #
    #     node.init_box.draw(rgb_image, (255, 0, 32), 3)
    #     cv.putText(rgb_image, info_text, (10, node.H - 40),
    #                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #
    #     # drown_image = node.bridge.cv2_to_compressed_imgmsg(rgb_image)
    #     # node.image_publisher.publish(drown_image)
    #     cv.imshow('frame', rgb_image)
    #     cv.waitKey(1)
    #
    # node.init_box_publisher.publish(ObjectBox(
    #     x1=node.init_box.x1,
    #     y1=node.init_box.y1,
    #     x2=node.init_box.x2,
    #     y2=node.init_box.y2
    # ))
    # rospy.loginfo('ok')
    # node.speaker_srv('Please let me see your back in 3 seconds')
    # front_image = inside_init_box[0].source_img
    # serialized_front_img = node.bridge.cv2_to_compressed_imgmsg(front_image)
    # front_descriptor = person_descriptor_extractor.parse_descriptor(front_image)
    # front_descriptor = front_descriptor[0].tolist()
    #
    # timeout = rospy.get_rostime() + init_timeout
    # while rospy.get_rostime() - timeout <= init_timeout:
    #     inside_init_box = node.inside_init_box
    #     if len(inside_init_box) != 1:
    #         timeout = rospy.get_rostime() + init_timeout
    #         if len(inside_init_box) == 0:
    #             info_text = 'Please stand inside the blue box'
    #         elif len(inside_init_box) > 1:
    #             info_text = 'Only 1 person can stand inside the blue box'
    #     else:
    #         info_text = 'Please turn back'
    #
    #     rgb_image = node.rgb_image
    #     node.init_box.draw(rgb_image, (255, 0, 32), 3)
    #
    #     cv.putText(rgb_image, info_text, (10, node.H - 40),
    #                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    #
    #     # drown_image = node.bridge.cv2_to_compressed_imgmsg(rgb_image)
    #     # node.image_publisher.publish(drown_image)
    #     cv.imshow('frame', rgb_image)
    #     cv.waitKey(1)
    #
    # back_image = inside_init_box[0].source_img
    # serialized_back_img = node.bridge.cv2_to_compressed_imgmsg(back_image)
    # back_descriptor = person_descriptor_extractor.parse_descriptor(back_image)
    # back_descriptor = back_descriptor[0].tolist()
    #
    # request_srv = PFInitializerRequest()
    # request_srv.front_img = serialized_front_img
    # request_srv.back_img = serialized_back_img
    # request_srv.front_descriptor = front_descriptor
    # request_srv.back_descriptor = back_descriptor
    #
    # response = node.call_person_follower(request_srv)
    # rospy.set_param('~initialized', True)
    # node.speaker_srv("I've remembered you, you can start walking")
