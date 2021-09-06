#!/usr/bin/env python3
from copy import copy

import cv2 as cv
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes
import message_filters


class ShowFaceDetections:
    MASK_VALID_COLOR = (32, 255, 0)
    MASK_INVALID_COLOR = (32, 0, 255)

    MALE_COLOR = (255, 100, 32)
    FEMALE_COLOR = (32, 10, 255)

    def __init__(self):
        self.mask_sub = message_filters.Subscriber(
            '/FMD/mask_is_on',
            ObjectBoxes,
        )
        self.gender_age_sub = message_filters.Subscriber(
            '/GAD/details',
            ObjectBoxes,
        )

        ts = message_filters.TimeSynchronizer([self.mask_sub, self.gender_age_sub], 10)
        ts.registerCallback(self.callback)

        self.bridge = CvBridge()
        self.dets = None

        self.main()

    def callback(self, mask_faces: ObjectBoxes, gender_age_faces: ObjectBoxes):
        self.dets = [mask_faces, gender_age_faces]

    def main(self):
        while not rospy.is_shutdown():
            if self.dets is None:
                continue

            face_dets, gender_age_faces = copy(self.dets)
            srcframe = self.bridge.compressed_imgmsg_to_cv2(face_dets.source_img)
            for mask_face, GA_faces in zip(face_dets.boxes, gender_age_faces.boxes):
                mask_on = bool(int(mask_face.label))
                gender, age = list(map(int, GA_faces.label.split(':')[:2]))

                gender_txt = 'Male' if gender == 1 else 'Female'
                label_txt = 'Mask off'

                gender_color = ShowFaceDetections.MALE_COLOR if gender == 1 else ShowFaceDetections.FEMALE_COLOR

                if mask_on:
                    mask_color = ShowFaceDetections.MASK_VALID_COLOR
                    label_txt = 'Mask on'
                else:
                    mask_color = ShowFaceDetections.MASK_INVALID_COLOR

                cv.putText(srcframe, gender_txt, (mask_face.x1, mask_face.y1 - 23), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                           gender_color, 2)
                cv.putText(srcframe, label_txt, (mask_face.x1, mask_face.y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                           mask_color, 2)

                half_h = (mask_face.y2 - mask_face.y1) // 2
                half_y = mask_face.y1 + half_h

                cv.line(srcframe, (mask_face.x1, mask_face.y1), (mask_face.x2, mask_face.y1), mask_color, 2)
                cv.line(srcframe, (mask_face.x1, mask_face.y1), (mask_face.x1, half_y), mask_color, 2)
                cv.line(srcframe, (mask_face.x2, mask_face.y1), (mask_face.x2, half_y), mask_color, 2)

                cv.line(srcframe, (mask_face.x1, half_y), (mask_face.x1, mask_face.y2), gender_color, 2)
                cv.line(srcframe, (mask_face.x1, mask_face.y2), (mask_face.x2, mask_face.y2), gender_color, 2)
                cv.line(srcframe, (mask_face.x2, half_y), (mask_face.x2, mask_face.y2), gender_color, 2)

            cv.imshow('frame', srcframe)
            key = cv.waitKey(1) & 0XFF
            if key in [27, ord('q')]:
                break


if __name__ == '__main__':
    rospy.init_node('show_mask_detection')
    node = ShowFaceDetections()
