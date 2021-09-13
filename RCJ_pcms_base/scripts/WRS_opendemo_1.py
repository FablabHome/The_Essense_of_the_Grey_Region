#!/usr/bin/env python3
from collections import deque

import rospy
from actionlib_msgs.msg import GoalID
from numpy import pi
from pyzbar import pyzbar
from robot_vision_msgs.msg import HumanPoses
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int16

from core.Nodes import MainProgram
from core.tools import ManipulatorController


class MainProg(MainProgram):
    def __init__(self):
        self.codes = {
            'alpha': self._alpha,
            'beta': self._beta,
            'delta': self._delta,
            'menu': self._menu,
        }

        super().__init__()
        # self._alpha()

    def _alpha(self):
        hand_queue = deque([False], maxlen=4)
        while not all(hand_queue):
            hand_queue.append(rospy.wait_for_message('/lift_hand_up_detection/hand_up_count', Int16).data > 0)

        self.speaker.say("I've saw you, coming")
        self.slam.go_to_loc('bedroom', wait_until_end=True)
        self.speaker.say('What do you want, mister?', wait_until_end=True)

    def _beta(self):
        # hand_queue = [False]
        # while not all(hand_queue):
        #     hand_queue.append(rospy.wait_for_message('/lift_hand_up_detection/hand_up_count', Int16).data > 0)
        #     if len(hand_queue) > 3:
        #         hand_queue.pop(0)
        #
        # self.speaker_pub.publish("I've saw you, coming")
        # data = {'point': [1.754736907, -1.976775948, -0.070871873, 0.99748542723], 'wait_until_end': True}
        # go_to_point.main(data, self.goal_pub)
        # self.speaker('May i take your order?')
        stats_queue = deque([False], maxlen=20)
        while not all(stats_queue):
            poses = rospy.wait_for_message('/pose_recognition/stats', HumanPoses).poses
            stats = list(filter(lambda p: p.pose == 'danger', poses))
            stats_queue.extend(stats[0:1])
            if len(stats_queue) == 0:
                stats_queue.append(False)

        self.cancel_slam.publish(GoalID())
        for _ in range(3):
            self.speaker.say('Somebody has fallen, please help', wait_until_end=True)

    def _delta(self):
        self.manipulator.set_joints(*ManipulatorController.INIT_JOINT, 1)
        self.manipulator.set_gripper(0.01, 1)
        rospy.sleep(2.5)

        self.manipulator.set_gripper(-0.01, 1)
        rospy.sleep(1)
        self.manipulator.set_joints(*ManipulatorController.HOME_JOINT, 1)
        rospy.sleep(1)
        self.manipulator.move_to(*ManipulatorController.HOME_POSE[:2], 0.137, 1)
        rospy.sleep(1)

        self.manipulator.set_joints(pi / 2, -1.601, 1.374, 0.238, 2)
        rospy.sleep(1.5)

        self.slam.go_to_loc('trash-cans', wait_until_end=True)
        turned = False
        tx, ty = 0, 0
        cx, cy = 0, 0
        while not rospy.is_shutdown():
            srcframe = rospy.wait_for_message('/bottom_camera_rotate/rgb/image_raw/compressed', CompressedImage)
            srcdepth = rospy.wait_for_message('/bottom_camera_rotate/depth/image_raw', Image)

            frame = self.bridge.compressed_imgmsg_to_cv2(srcframe)
            depth = self.bridge.imgmsg_to_cv2(srcdepth)

            try:
                barcode = pyzbar.decode(frame)[0]
                x, y, w, h = barcode.rect
                cx = x + w // 2
                cy = y + h // 2
                tx, ty = cx, cy
            except IndexError:
                rospy.loginfo('hi')
                cx, cy = tx, ty

            cz = depth[cy, cx]
            rospy.loginfo(f'{cx}, {cz}')
            if cx == 0:
                continue

            if abs(cx) - 320 > 15 and not turned:
                az = -0.06 if cx > 20 else 0.06
                self.chassis.move(0, az)
            else:
                turned = True
                if cz == 0:
                    continue

                if cz - 905 > 4:
                    self.chassis.move(0.1, 0)
                else:
                    break

        self.manipulator.set_joints(0, -1.601, 1.374, 0.238, 2)
        rospy.sleep(1.5)

        self.manipulator.set_joints(*ManipulatorController.INIT_JOINT, 1)
        rospy.sleep(1)
        self.manipulator.move_to(0.33, *ManipulatorController.INIT_POSE[-2:], 1)
        rospy.sleep(1)
        self.manipulator.move_to(0.35, 0, 0.182, 1)
        rospy.sleep(1)
        self.manipulator.set_gripper(0.01, 1)
        rospy.sleep(1)

        self.manipulator.set_joints(*ManipulatorController.HOME_JOINT, 1)
        rospy.sleep(1)
        self.manipulator.move_to(*ManipulatorController.HOME_POSE[:2], 0.137, 1)
        rospy.sleep(1)
        self.manipulator.set_joints(pi / 2, -1.601, 1.374, 0.238, 2)
        rospy.sleep(1.5)

        self.chassis.move(-0.2, 0)
        rospy.sleep(1.2)
        self.chassis.move(-0.2, 0)
        rospy.sleep(1.2)

        self.slam.go_to_loc('bedroom', wait_until_end=True)
        self.speaker.say("I've thrown your bottle", wait_until_end=True)

    def _menu(self):
        # self.facial_pub.publish('menu:')
        pass


if __name__ == '__main__':
    rospy.init_node('main_prog')
    node = MainProg()
    rospy.spin()
