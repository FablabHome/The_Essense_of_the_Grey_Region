#!/usr/bin/env python3
from collections import deque

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from home_robot_msgs.msg import ObjectBoxes
from mr_voice.srv import SpeakerSrv
from rospkg import RosPack
from std_msgs.msg import String, Int16, Empty
from action_commands import go_to_point


class MainProg:
    def __init__(self):
        self.codes = {
            'alpha': self._alpha,
            'beta': self._beta,
            'delta': self._delta,
            'final': self._final,
        }

        self.speaker_srv = rospy.ServiceProxy('/speaker/text', SpeakerSrv)
        self.play_pub = rospy.Publisher(
            '/birthday_song/play',
            Empty,
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
        data = {'point': [x, y, z, w], 'wait_until_end': wait_until_end}
        go_to_point.main(data, self.goal_pub)

    def _alpha(self):
        self.__go_to_point(1.48625732466, -1.87237822643, -0.723484843933, 0.690340264362, wait_until_end=True)

        rospy.set_param('/YD/lock', False)
        rospy.set_param('/manipulator_grab/lock', False)

        while not rospy.get_param('/manipulator_grab/grabbed'):
            continue

        t = Twist()
        t.linear.x = -0.2
        self.wheel_pub.publish(t)
        rospy.sleep(2)

        self.__go_to_point(2.45894356306, -1.69283733473, -0.789928684276, 0.613198722894, wait_until_end=True)
        rospy.set_param('/manipulator_grab/place', True)
        while not rospy.get_param('/manipulator_grab/placed'):
            continue

        self.__go_to_point(2.87856819597, 0.0795860457531, -0.994349938809, 0.106151774318, wait_until_end=True)

    def _beta(self):
        hand_queue = [False]
        while not all(hand_queue):
            hand_queue.append(rospy.wait_for_message('/lift_hand_up_detection/hand_up_count', Int16).data > 0)
            if len(hand_queue) > 3:
                hand_queue.pop(0)

        self.speaker_pub.publish("I've saw you, coming")
        data = {'point': [1.754736907, -1.976775948, -0.070871873, 0.99748542723], 'wait_until_end': True}
        go_to_point.main(data, self.goal_pub)
        self.speaker_srv('May i take your order?')

    def _delta(self):
        rospy.set_param('/FMD/kill', True)
        rospy.set_param('/YD/lock', False)
        t = Twist()

        rospy.set_param('/manipulator_grab/lock', False)

        while not rospy.get_param('/manipulator_grab/grabbed'):
            continue

        rospy.set_param('/YD/lock', True)
        rospy.sleep(3)

        t.linear.x = -0.1
        self.wheel_pub.publish(t)

        rospy.sleep(1.5)

        t.linear.x = -0.25
        self.wheel_pub.publish(t)
        rospy.sleep(2)

        data = {'point': [1.754736907, -1.976775948, 0.99742867, 0.07166617], 'wait_until_end': True}
        go_to_point.main(data, self.goal_pub)
        self.speaker_srv('Here are your drinks, please enjoy')

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
