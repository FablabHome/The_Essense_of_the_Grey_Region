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
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from .Abstract import Tools


class Chassis(Tools):
    def __init__(self, cmd_topic='/cmd_vel'):
        super()._check_status()

        self.cmd_topic = cmd_topic
        self.imu = None
        self.twist_publisher = rospy.Publisher(
            self.cmd_topic,
            Twist,
            queue_size=1
        )

        self.twist = Twist()

        self.sub = rospy.Subscriber(
            "/imu/data",
            Imu,
            self.imu_callback, queue_size=1
        )

    def imu_callback(self, msg: Imu):
        self.imu = msg

    def move(self, forward_speed, turn_speed):
        self.twist.linear.x = forward_speed
        self.twist.angular.z = turn_speed

        self.twist_publisher.publish(self.twist)

    def turn(self, angle: float):
        q = [
            self.imu.orientation.x,
            self.imu.orientation.y,
            self.imu.orientation.z,
            self.imu.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(q)
        target = yaw + angle
        if target > np.pi:
            target = target - np.pi * 2
        elif target < -np.pi:
            target = target + np.pi * 2
        self.__turn_to(target, 0.25)

    def check_yaw(self):
        while True:
            q = [
                self.imu.orientation.x,
                self.imu.orientation.y,
                self.imu.orientation.z,
                self.imu.orientation.w
            ]
            roll, pitch, yaw = euler_from_quaternion(q)
            print(yaw)

    def __turn_to(self, angle: float, speed: float):
        max_speed = 0.5
        limit_time = 20
        start_time = rospy.get_time()
        while True:
            q = [
                self.imu.orientation.x,
                self.imu.orientation.y,
                self.imu.orientation.z,
                self.imu.orientation.w
            ]
            roll, pitch, yaw = euler_from_quaternion(q)
            e = -(angle - yaw)
            # if yaw > 0 and angle < 0:
            # cw = np.pi + yaw + np.pi - angle
            # aw = -yaw + angle
            # if cw < aw:
            #     e = -cw
            # elif yaw < 0 and angle > 0:
            #     cw = yaw - angle
            #     aw = np.pi - yaw + np.pi + angle
            #     if aw < cw:
            #         e = aw
            print("e:", e, "yaw:", yaw, "angle:", angle)
            if abs(e) < 0.01 or rospy.get_time() - start_time > limit_time:
                print("OK")
                break
            self.move(0.0, max(min(max_speed, speed * e), 0.2))
            rospy.Rate(20).sleep()
        self.move(0.0, 0.0)
