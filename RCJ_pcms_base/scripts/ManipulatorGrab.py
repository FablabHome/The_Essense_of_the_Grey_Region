#!/usr/bin/env python3
from copy import copy
from math import pi

import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from home_robot_msgs.msg import ObjectBoxes
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest, SetJointPosition, \
    SetJointPositionRequest
from sensor_msgs.msg import Image
from std_msgs.msg import String

from core.Nodes import Node


class ManipulatorGrab(Node):
    MANI_SRV_NAME = '/goal_task_space_path_position_only'
    MANI_GRIPPER_SRV_NAME = '/goal_tool_control'
    MANI_JOINT_SRV_NAME = '/goal_joint_space_path'

    MANI_HEIGHT_ERR = 26
    MANI_FRONT_ERR = 28
    INIT_TIMEOUT = 1
    FOV_H = 60
    FOV_V = 49.5

    CAMERA_ANGLE = -16

    W = 640
    H = 480

    def __init__(self):
        super(ManipulatorGrab, self).__init__('manipulator_grab')

        rospy.wait_for_service(ManipulatorGrab.MANI_SRV_NAME)
        self.service = rospy.ServiceProxy(ManipulatorGrab.MANI_SRV_NAME, SetKinematicsPose)

        rospy.wait_for_service(ManipulatorGrab.MANI_GRIPPER_SRV_NAME)
        self.gripper_control = rospy.ServiceProxy(ManipulatorGrab.MANI_GRIPPER_SRV_NAME, SetJointPosition)

        rospy.wait_for_service(ManipulatorGrab.MANI_JOINT_SRV_NAME)
        self.joint_control = rospy.ServiceProxy(ManipulatorGrab.MANI_JOINT_SRV_NAME, SetJointPosition)

        self.bridge = CvBridge()

        rospy.set_param('~grabbed', False)
        rospy.set_param('~lock', True)
        rospy.set_param('~place', False)
        rospy.set_param('~placed', False)

        self.twist_pub = rospy.Publisher(
            '/cmd_vel',
            Twist,
            queue_size=1
        )

        self.speaker_pub = rospy.Publisher(
            '/speaker/say',
            String,
            queue_size=1
        )

        self.yolo_sub = rospy.Subscriber(
            '/YD/boxes',
            ObjectBoxes,
            self.box_callback,
            queue_size=1
        )

        self.srcframe = None
        self.bottle_box = None
        self.main()

    def box_callback(self, boxes: ObjectBoxes):
        detection_boxes = boxes.boxes
        key_func = lambda x: x.label.strip() == 'bottle' and (x.y2 - x.y1) * (x.x2 - x.x1) >= 500
        bottle_boxes = list(filter(key_func, detection_boxes))
        bottle_boxes = sorted(bottle_boxes, key=lambda b: b.x1)
        self.srcframe = self.bridge.compressed_imgmsg_to_cv2(boxes.source_img)
        if len(bottle_boxes) > 0:
            self.bottle_box = bottle_boxes[-1]
        else:
            self.bottle_box = None

    def move_to(self, x, y, z, t):
        try:
            req = SetKinematicsPoseRequest()
            req.end_effector_name = 'gripper'
            req.kinematics_pose.pose.position.x = x
            req.kinematics_pose.pose.position.y = y
            req.kinematics_pose.pose.position.z = z
            req.path_time = t

            resp = self.service(req)
            return resp
        except Exception as e:
            rospy.loginfo(e)
            return False

    def set_gripper(self, position, t):
        try:
            req = SetJointPositionRequest()
            req.joint_position.joint_name = ["gripper"]
            req.joint_position.position = [position]
            req.path_time = t
            resp = self.gripper_control(req)
            return resp
        except Exception as e:
            rospy.loginfo(e)

    def set_joints(self, joint1, joint2, joint3, joint4, t):
        try:
            req = SetJointPositionRequest()
            req.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
            req.joint_position.position = [joint1, joint2, joint3, joint4]
            req.path_time = t
            resp = self.joint_control(req)
            return resp
        except Exception as e:
            rospy.loginfo(e)

    @staticmethod
    def angle_2_radian(angle):
        return (np.pi * angle) / 180

    def real_xyz(self, x, y, z):
        rad_h = self.angle_2_radian(ManipulatorGrab.FOV_H / 2)
        rad_v = self.angle_2_radian(ManipulatorGrab.FOV_V / 2)
        rad_cam_angle = self.angle_2_radian(ManipulatorGrab.CAMERA_ANGLE)
        real_w = 2 * z * np.tan(rad_h)
        real_h = 2 * z * np.tan(rad_v)

        # Real x
        real_x = real_w * x / ManipulatorGrab.W
        real_x -= (real_w / 2)

        # Real y
        reality_y = real_h * y / ManipulatorGrab.H
        FE = z * np.sin(rad_cam_angle)
        GF = (0.5 * real_h - reality_y) * np.cos(rad_cam_angle)
        real_y = FE + GF

        # Real z
        OD = z * np.cos(rad_cam_angle)
        ED = GF * np.tan(rad_cam_angle)
        real_z = OD - ED

        return real_x, real_y, real_z

    @staticmethod
    def __avoid_zeropoints(point, depth_image, limit=None):
        if limit is None:
            limit = depth_image.shape[0]

        x, y = point
        if depth_image[y, x] != 0:
            return depth_image[y, x]

        for each in range(1, limit):
            up = y - each
            down = y + each
            left_x = x - each
            right_x = x + each

            top = depth_image[up:up + 1, left_x:right_x + 1]
            left = depth_image[up:up + 1, left_x:left_x + 1]
            bottom = depth_image[down:down + 1, left_x:left_x + 1]
            right = depth_image[up:down + 1, right_x:right_x + 1]

            for block in [top, left, bottom, right]:
                nonzero = block[np.nonzero(block)]
                if nonzero.shape[0] > 0:
                    distance = nonzero[0]
                    return distance

        return -1

    # def find_bottle(self, status_list, start_time):
    #
    #     return box

    def main(self):
        lower = np.array([22, 93, 0], dtype="uint8")
        upper = np.array([45, 255, 255], dtype="uint8")
        while rospy.get_param('~lock'):
            continue

        self.move_to(0.149, 0.0, 0.149, 1.5)
        rospy.sleep(1.5)
        self.move_to(0.145, 0.0, 0.140, 1.5)
        rospy.sleep(1.5)
        self.move_to(0.138, 0.0, 0.132, 1.5)
        rospy.sleep(1.5)
        self.move_to(0.130, 0.0, 0.126, 1.5)
        rospy.sleep(1.5)

        self.set_joints(pi / 2, -1.601, 1.374, 0.238, 2)
        rospy.sleep(1.5)

        self.set_gripper(0.01, 2)
        rospy.sleep(1.5)

        try:
            start_time = rospy.get_rostime() + rospy.Duration(ManipulatorGrab.INIT_TIMEOUT)
            status_list = [False] * 20
            box = copy(self.bottle_box)
            while rospy.get_rostime() - start_time <= rospy.Duration(0) or not all(status_list):
                # rospy.loginfo(len(self.bottle_boxes))
                # if len(self.bottle_boxes) != 1 and status_list.count(False) >= 10:
                #     status_list.append(False)
                #     start_time = rospy.get_rostime() + rospy.Duration(ManipulatorGrab.INIT_TIMEOUT)
                # status_list.append(True)
                # if len(status_list) > 20:

                #     status_list.pop(0)
                if len(status_list) > 20:
                    status_list.pop(0)

                if self.bottle_box is None:
                    status_list.append(False)
                    start_time = rospy.get_rostime() + rospy.Duration(ManipulatorGrab.INIT_TIMEOUT)
                    continue

                status_list.append(True)
                box = copy(self.bottle_box)
                rospy.loginfo(status_list)

            # self.speaker_pub.publish(String("Please don't move the bottle"))

            cx = cy = .0
            last_cz = cz = 0
            last_box = copy(box)
            turned = False
            while not rospy.is_shutdown():
                box = copy(self.bottle_box)
                if box is None:
                    box = copy(last_box)

                cx = box.x1 + (box.x2 - box.x1) // 2
                cy = box.y1 + (box.y2 - box.y1) // 2

                depth_img = rospy.wait_for_message('/bottom_camera/depth/image_raw', Image)
                depth_img = self.bridge.imgmsg_to_cv2(depth_img)
                cz = self.__avoid_zeropoints((cx, cy), depth_img, limit=20)
                # cz = depth_img[cy, cx]
                rospy.loginfo(f'\t\t{cz}')
                cz = cz if cz != -1 else last_cz
                if cz == 0: continue
                rx, ry, rz = self.real_xyz(cx, cy, cz)
                rx -= 15

                t = Twist()
                rospy.loginfo(f'\t{rx}, {rz}')

                if abs(rx) > 30 and not turned:
                    t.angular.z = 0.02 if rx > 20 else -0.02
                    self.twist_pub.publish(t)
                else:
                    turned = True
                    if rz - 570 > 4:
                        rospy.loginfo('test')
                        t.linear.x = 0.05
                        self.twist_pub.publish(t)
                    else:
                        break

                last_cz = cz
                last_box = copy(box)

            # self.yolo_sub.unregister()
            # cx = box.x1 + (box.x2 - box.x1) // 2
            # cy = box.y1 + (box.y2 - box.y1) // 2
            # get_depth_img = rospy.wait_for_message('/bottom_camera/depth/image_raw', Image)
            # depth_img = self.bridge.imgmsg_to_cv2(get_depth_img)
            # cz = self.__avoid_zeropoints((cx, cy), depth_img)
            print(cx, cy, cz)

            x, _, z = self.real_xyz(cx, cy, cz)

            z -= ManipulatorGrab.MANI_FRONT_ERR * 10

            z /= 1000
            x /= 1000
            y = .15

            z = round(z, 2)

            self.set_joints(0, -1.601, 1.374, 0.238, 2)
            rospy.sleep(2)

            print(z, x, y)
            self.move_to(0.1, 0, y, 2)
            rospy.sleep(2)
            self.move_to(z / 3, 0, y, 1)
            rospy.sleep(1)
            self.move_to(z / 2, 0, y, 1)
            rospy.sleep(1)
            self.move_to(z, 0, y, 2)
            rospy.sleep(2)
            # self.move_to(z, 0, y, 1)
            # rospy.sleep(1)
            # self.move_to(z + 0.02, 0, y, 1)
            # rospy.sleep(1)
            # self.move_to(z * 0.9, 0, .15, 1)
            # rospy.sleep(1)
            # self.move_to(z * 0.9, 0, .11, 1)
            # rospy.sleep(1)
            # self.move_to(z * 0.9, 0, .09, 1)
            # rospy.sleep(1)

        except Exception as e:
            rospy.loginfo(e)
            self.speaker_pub.publish('Please land me the bottle in 5 seconds')
            self.move_to(0.287, 0, 0.191, 2)
            rospy.sleep(7)
            self.speaker_pub.publish('Closing the gripper')

        # # Moving back
        # self.set_gripper(-0.01, 2)
        # rospy.sleep(2)
        #
        # self.move_to(0.135, 0, 0.238, 1)
        # rospy.sleep(1)
        #
        # t = Twist()
        # t.linear.x = -0.2
        # self.twist_pub.publish(t)
        # rospy.sleep(2)

        # Put it on shelve
        self.set_gripper(-0.01, 2)
        rospy.sleep(2)

        self.move_to(0.19, 0.0, 0.2, 1)
        rospy.sleep(1)

        t = Twist()
        t.linear.x = -0.2
        self.twist_pub.publish(t)
        rospy.sleep(2)

        self.move_to(0.13, 0.0, 0.154, 1)
        rospy.sleep(1)
        self.move_to(0.13, 0.0, 0.114, 1)
        rospy.sleep(1)

        self.move_to(0.130, 0.0, 0.126, 1.5)
        rospy.sleep(1.5)

        self.set_joints(pi / 2, -1.601, 1.374, 0.238, 2)
        rospy.sleep(1.5)

        # self.move_to(0.135, 0, 0.238, 1)
        # rospy.sleep(1)
        #
        # self.move_to(0.13, 0.0, 0.144, 1)
        # rospy.sleep(1)
        # self.move_to(0.13, 0.0, 0.124, 1)
        # rospy.sleep(1)
        # self.move_to(0.13, 0.0, 0.114, 1)
        # rospy.sleep(1)
        #
        # self.set_joints(pi, -1.652, 1.473, 0.212, 2)
        # rospy.sleep(2)
        #
        # self.set_gripper(0.01, 2)
        # rospy.sleep(2)
        #
        # self.set_joints(pi, -1.652, 0.949, 0.212, 2)
        # rospy.sleep(2)
        #
        # self.set_joints(pi / 2, -1.652, 0.949, 0.212, 2)
        # rospy.sleep(2)

        rospy.set_param('~grabbed', True)
        while not rospy.get_param('~place'):
            continue

        while not rospy.is_shutdown():
            if self.srcframe is None:
                continue

            frame = self.srcframe.copy()
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, lower, upper)
            cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            # for c in cnts:
            #     x, y, w, h = cv.boundingRect(c)
            #     if w * h > 1000:
            #         print(w * h)
            #         cv.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
            boxes = list(map(cv.boundingRect, cnts))
            # rospy.loginfo(boxes)
            boxes = list(filter(lambda b: b[2] * b[3] > 5000, boxes))
            if len(boxes) == 0: continue
            pikachu = max(boxes, key=lambda a: a[2] * a[3])
            x, y, w, h = pikachu
            cx = x + w // 2
            cy = x + h // 2

            depth_img = rospy.wait_for_message('/bottom_camera/depth/image_raw', Image)
            depth_img = self.bridge.imgmsg_to_cv2(depth_img)
            cz = self.__avoid_zeropoints((cx, cy), depth_img, limit=20)
            rospy.loginfo(rz)
            rx, ry, rz = self.real_xyz(cx, cy, cz)
            t = Twist()
            if rz - 817 > 4:
                rospy.loginfo('test')
                t.linear.x = 0.05
                self.twist_pub.publish(t)
            else:
                break

        self.set_joints(0, -1.601, 1.374, 0.238, 2)
        rospy.sleep(2)
        self.move_to(0.13, 0.0, 0.114, 1.1)
        rospy.sleep(1.1)
        self.move_to(0.13, 0.0, 0.134, 1.1)
        rospy.sleep(1.1)
        self.move_to(0.13, 0.0, 0.164, 1.1)
        rospy.sleep(1.1)
        self.move_to(0.13, 0.0, 0.184, 1.1)
        rospy.sleep(1.1)
        self.move_to(0.13, 0.0, 0.2, 0.7)
        rospy.sleep(0.7)
        self.move_to(0.2, 0.0, 0.23, 1)
        rospy.sleep(1)
        self.move_to(0.26, 0.0, 0.26, 1)
        rospy.sleep(1)
        self.set_gripper(0.01, 2)
        rospy.sleep(2)
        rospy.set_param('~placed', True)

        '''
        On manipulator:
        x = z
        y = x
        z = y
        '''

    def reset(self):
        pass


if __name__ == '__main__':
    node = ManipulatorGrab()
