#!/usr/bin/env python3
from collections import deque

import rospy
from actionlib_msgs.msg import GoalID
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from mr_voice.srv import SpeakerSrv
from numpy import pi
from robot_vision_msgs.msg import HumanPoses
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String, Int16, Empty

from action_commands import go_to_point
from core.tools import ManipulatorController, Chassis

from pyzbar import pyzbar


class MainProg:
    def __init__(self):
        self.codes = {
            'alpha': self._alpha,
            'beta': self._beta,
            'delta': self._delta,
            'menu': self._menu,
            'final': self._final,
        }

        self.speaker_srv = rospy.ServiceProxy('/speaker/text', SpeakerSrv)
        self.manipulator = ManipulatorController()
        self.chassis = Chassis()
        self.bridge = CvBridge()

        self.play_pub = rospy.Publisher(
            '/birthday_song/play',
            Empty,
            queue_size=1
        )
        self.cancel_slam = rospy.Publisher(
            '/move_base/cancel',
            GoalID,
            queue_size=1
        )
        self.pause_pub = rospy.Publisher(
            '/birthday_song/pause',
            Empty,
            queue_size=1
        )
        self.speaker_pub = rospy.Publisher(
            '/speaker/say',
            String,
            queue_size=1
        )
        self.wheel_pub = rospy.Publisher(
            '/cmd_vel',
            Twist,
            queue_size=1
        )
        self.goal_pub = rospy.Publisher(
            '/move_base_simple/goal',
            PoseStamped,
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

        self.base = RosPack().get_path('rcj_pcms_base')

        self._alpha()

    def code_callback(self, data):
        code = data.data
        self.codes[code]()

    def __go_to_point(self, x, y, z, w, wait_until_end=False):
        data = {'point': [x, y, z, w], 'loc': None, 'wait_until_end': wait_until_end}
        go_to_point.main(data, self.goal_pub)

    def __go_to_loc(self, loc, wait_until_end=False):
        data = {'point': None, 'loc': loc, 'wait_until_end': wait_until_end}
        go_to_point.main(data, self.goal_pub)

    def _alpha(self):
        hand_queue = deque([False], maxlen=4)
        while not all(hand_queue):
            hand_queue.append(rospy.wait_for_message('/lift_hand_up_detection/hand_up_count', Int16).data > 0)

        self.speaker_pub.publish("I've saw you, coming")
        self.__go_to_loc('bedroom', wait_until_end=True)
        self.speaker_srv('What do you want, mister?')

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
        # self.speaker_srv('May i take your order?')
        stats_queue = deque([False], maxlen=20)
        while not all(stats_queue):
            poses = rospy.wait_for_message('/pose_recognition/stats', HumanPoses).poses
            stats = list(filter(lambda p: p.pose == 'danger', poses))
            stats_queue.extend(stats[0:1])
            if len(stats_queue) == 0:
                stats_queue.append(False)

        self.cancel_slam.publish(GoalID())
        for _ in range(3):
            self.speaker_srv('Somebody has fallen, please help')

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

        self.__go_to_loc('trash-cans', wait_until_end=True)
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

        self.__go_to_loc('bedroom', wait_until_end=True)
        self.speaker_srv("I've thrown your bottle")

    def _menu(self):
        # self.facial_pub.publish('menu:')
        pass

    def _final(self):
        self.play_pub.publish()
        data = {'point': [-1.58027, 0.104909, 0.1022311, 0.9947606], 'wait_until_end': True}
        go_to_point.main(data, self.goal_pub)
        rospy.sleep(3)
        self.pause_pub.publish()
        self.speaker_srv('I will help you take a precious picture')
        rospy.sleep(2)
        self.speaker_srv('Aurora, you look absolutely gorgeous. Would u please move a little to the left')
        rospy.sleep(4)
        self.speaker_srv('Three, two, one cheese')
        rospy.sleep(4)
        self.speaker_srv('Thank you Mister and Miss, hope you enjoy your meal. Have a wonderful night')


if __name__ == '__main__':
    rospy.init_node('main_prog')
    node = MainProg()
    rospy.spin()
