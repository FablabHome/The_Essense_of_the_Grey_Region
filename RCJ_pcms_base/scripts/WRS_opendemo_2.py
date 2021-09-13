#!/usr/bin/env python3
from collections import deque

import requests
import rospy
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import PoseWithCovarianceStamped
from robot_vision_msgs.msg import HumanPoses
from std_srvs.srv import SetBoolRequest, SetBool
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from core.Nodes import MainProgram


class WRSDemo2(MainProgram):
    def __init__(self):
        self.codes = {
            'weather': self._weather,
            'grab': self._grab,
            'help': self._help
        }
        super().__init__()
        self.bottles = {
            'green': 0,
            'blue': 1,
            'purple': -1
        }

        self.kill_yolo = rospy.ServiceProxy('/YD/kill', SetBool)

    def _weather(self):
        appid = 'c73d9cdb31fd6a386bee66158b116cd0'
        city = 'macau'
        resp = requests.get(f'https://api.openweathermap.org/data/2.5/weather?appid={appid}&q={city}&units=metric')
        data = resp.json()
        weather = data['main']['temp']
        self.speaker.say(
            f"It’s mostly clear at {int(weather)} degrees, I suggest you go out to have a walk, it will be amazing")

    def _grab(self):
        rospy.set_param('/YD/lock', False)
        resp = requests.get('https://std.puiching.edu.mo/~0763236-3/cgi-bin/WRS_color_write.py?drink=black')
        bottle = resp.json()['drink']
        self.speaker.say(f"You guys have voted the {bottle} drink")
        rospy.set_param('/manipulator_grab/bottle', self.bottles[bottle])
        rospy.set_param('/manipulator_grab/lock', False)
        while not rospy.get_param('/manipulator_grab/grabbed'):
            continue

        self.kill_yolo(SetBoolRequest(True))
        self.slam.go_to_loc('bedroom', wait_until_end=True)

        rospy.sleep(3)
        rospy.set_param('/FMD/lock', False)
        self.speaker.say(
            'Here’s your drink, you can just put it on my shelf. As we are going out, you can put all your belongings on my back.')

    def _help(self):
        self.cancel_slam.publish(GoalID())
        # amcl_pose = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped).pose.pose
        # position = amcl_pose.position
        # orientation = amcl_pose.orientation
        #
        # x, y = position.x, position.y
        # z, w = orientation.z, orientation.w
        #
        # az = euler_from_quaternion(0, 0, z, w) + 3.14
        # new_z, new_w = quaternion_from_euler(0, 0, az)
        #
        # self.slam.go_to_point(x, y, new_z, new_w, wait_until_end=True)
        self.speaker.say('Are you alright?')
        stats_queue = deque([False], maxlen=20)
        while not all(stats_queue):
            poses = rospy.wait_for_message('/pose_recognition/stats', HumanPoses).poses
            stats = list(filter(lambda p: p.pose == 'danger', poses))
            stats_queue.extend(stats[0:1])
            if len(stats_queue) == 0:
                stats_queue.append(False)

        for _ in range(3):
            self.speaker.say('Somebody has fallen, please help', wait_until_end=True)


if __name__ == '__main__':
    rospy.init_node('main_prog')
    node = WRSDemo2()
    rospy.spin()
