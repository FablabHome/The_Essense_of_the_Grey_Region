#!/usr/bin/env python3
from os import path

import cv2 as cv
import matplotlib
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes
from rospkg import RosPack

from core.Detection import ZeroDCEEnhancement


def callback(boxes: ObjectBoxes):
    global persons, srcframe, bridge
    srcframe = bridge.compressed_imgmsg_to_cv2(boxes.source_img)
    persons = list(filter(lambda b: b.label.strip() == 'person', boxes.boxes))


persons = []
srcframe = None
if __name__ == '__main__':
    rospy.init_node('test_image_enhancement')
    base = RosPack().get_path('rcj_pcms_base') + '/..'
    model_path = path.join(base, 'models/Zero-DCE/Epoch99.pth')

    plt = matplotlib.pyplot
    matplotlib.use('TkAgg')

    bridge = CvBridge()
    rospy.Subscriber(
        '/YD/boxes',
        ObjectBoxes,
        callback,
        queue_size=1
    )
    # DCE_net = ZeroDCEEnhancement(model_path)
    plt.ion()
    plt.show()
    while not rospy.is_shutdown():
        if srcframe is None:
            continue

        frame = srcframe.copy()
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        axes = plt.gca()
        axes.set_ylim([0, 3000])
        plt.hist(value.ravel(), 256, [0, 256], alpha=0.5, label='background')

        if len(persons) > 0:
            person = persons[0]
            person_img = bridge.compressed_imgmsg_to_cv2(person.source_img)
            person_hsv = cv.cvtColor(person_img, cv.COLOR_BGR2HSV)
            person_value = person_hsv[:, :, 2]
            axes = plt.gca()
            axes.set_ylim([0, 3000])
            plt.hist(person_value.ravel(), 256, [0, 256], alpha=0.5, label='person')

        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(.001)
        plt.clf()
