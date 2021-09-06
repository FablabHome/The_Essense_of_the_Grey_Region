#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import Twist


class TestPIDData:
    def __init__(self):
        rospy.init_node('test_pid')
        rospy.Subscriber(
            '/mobile_base/commands/velocity',
            Twist,
            self.callback,
            queue_size=1
        )
        self.rate = rospy.Rate(30)

        self.recorded_forward_data = []
        self.recorded_turn_data = []

        self.main()

    def callback(self, data: Twist):
        self.recorded_forward_data.append(data.linear.x)
        self.recorded_turn_data.append(data.angular.z)

    def main(self):
        while not rospy.is_shutdown():
            self.rate.sleep()

        fig, axs = plt.subplots(2, 1)
        axs[0].set_yticklabels(np.arange(-1.2, 1.2, 0.2))
        axs[0].set_ylim(-1.2, 1.2)
        axs[0].plot(self.recorded_forward_data)
        axs[0].set_xlabel('sequence')
        axs[0].set_ylabel('forward_data')

        axs[1].set_yticklabels(np.arange(-1.2, 1.2, 0.2))
        axs[0].set_ylim(-1.2, 1.2)
        axs[1].plot(self.recorded_turn_data)
        axs[1].set_xlabel('sequence')
        axs[1].set_ylabel('turn_data')

        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    node = TestPIDData()
