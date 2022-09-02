#!/usr/bin/env python3
import math

import cv2
import mediapipe as mp
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse
from core.Nodes import Node
import imutils.video


class HWPFD(Node):
    HANDS_UP_THRESHOLD = 0.4
    FINGER_Y_THRESHOLD = 17

    def __init__(self, camera_topic):
        super(HWPFD, self).__init__('pose_single', anonymous=False)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh

        self.bridge = CvBridge()

        self.srcframe = None
        self.lock = False

        self.hands_up_pub = rospy.Publisher(
            '~hands_up',
            Bool,
            queue_size=1
        )

        self.finger_direction_pub = rospy.Publisher(
            '~finger_direction',
            String,
            queue_size=1
        )

        rospy.Subscriber(
            camera_topic,
            CompressedImage,
            self.callback,
            queue_size=1
        )

        rospy.Service('~lock', Trigger, self.lock_handler)

    def lock_handler(self, req):
        self.lock = not self.lock
        return TriggerResponse()

    def callback(self, image):
        self.srcframe = self.bridge.compressed_imgmsg_to_cv2(image)

    def get_landmark(self, landmark_idx, pose_landmarks, image):
        landmark = pose_landmarks.landmark[landmark_idx]
        return self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, *image.shape[1::-1])

    @staticmethod
    def euclidean_distance(landmark1, landmark2):
        return math.sqrt((landmark2[1] - landmark1[1]) ** 2 + (landmark2[0] - landmark1[0]) ** 2)

    @staticmethod
    def nNone(instance):
        return instance is not None

    @staticmethod
    def allnNone(*instance_list):
        for instance in instance_list:
            if instance is None:
                return False
        return True

    @staticmethod
    def law_of_cosine(d1, d2, bottom):
        return (d1 ** 2 + d2 ** 2 - bottom ** 2) / (2 * d1 * d2)

    def get_three_point_cos(self, pb1, pb2, pt1):
        ds1 = self.euclidean_distance(pb1, pt1)
        ds2 = self.euclidean_distance(pt1, pb2)
        db1 = self.euclidean_distance(pb1, pb2)
        if ds1 * ds2 * db1 == 0:
            return 0

        cos = self.law_of_cosine(ds1, ds2, db1)
        return cos

    def reset(self) -> TriggerResponse:
        return TriggerResponse()

    def main(self):
        # For webcam input:
        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            fps = imutils.video.FPS().start()
            while not rospy.is_shutdown():
                if self.srcframe is None or self.lock:
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.

                image = self.srcframe.copy()
                image = cv2.flip(image, 1)
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                H, *_ = image.shape

                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles
                    .get_default_pose_landmarks_style())

                self.mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                self.mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    self.mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Waving Detection
                if self.nNone(results.pose_landmarks):
                    shoulder_left = self.get_landmark(12, results.pose_landmarks, image)
                    shoulder_right = self.get_landmark(11, results.pose_landmarks, image)

                    elbow_left = self.get_landmark(14, results.pose_landmarks, image)
                    elbow_right = self.get_landmark(13, results.pose_landmarks, image)

                    wrist_left = self.get_landmark(16, results.pose_landmarks, image)
                    wrist_right = self.get_landmark(15, results.pose_landmarks, image)

                    left_up = False
                    right_up = False

                    if self.allnNone(shoulder_left, elbow_left, wrist_left):
                        cos_left = -self.get_three_point_cos(shoulder_left, wrist_left, elbow_left)
                        if elbow_left[1] <= shoulder_left[1] and cos_left >= HWPFD.HANDS_UP_THRESHOLD:
                            left_up = True
                        cv2.circle(image, shoulder_left, 17, (255, 0, 255), -1)
                        cv2.circle(image, elbow_left, 17, (255, 0, 0), -1)
                        cv2.circle(image, wrist_left, 17, (0, 0, 255), -1)

                    if self.allnNone(shoulder_right, elbow_right, wrist_right):
                        cos_right = -self.get_three_point_cos(shoulder_right, wrist_right, elbow_right)
                        if elbow_right[1] <= shoulder_right[1] and cos_right >= HWPFD.HANDS_UP_THRESHOLD:
                            right_up = True
                        cv2.circle(image, shoulder_right, 17, (255, 0, 255), -1)
                        cv2.circle(image, elbow_right, 17, (255, 0, 0), -1)
                        cv2.circle(image, wrist_right, 17, (0, 0, 255), -1)

                    if left_up or right_up:
                        self.hands_up_pub.publish(True)
                        cv2.putText(image, 'Hands Up', (10, H - ((2 * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Finger direction recognition
                hand_landmarks = results.left_hand_landmarks
                if self.nNone(hand_landmarks):
                    pass
                elif self.nNone(results.right_hand_landmarks):
                    hand_landmarks = results.right_hand_landmarks

                if self.nNone(hand_landmarks):
                    index_finger_1 = self.get_landmark(5, hand_landmarks, image)
                    index_finger_2 = self.get_landmark(6, hand_landmarks, image)
                    index_finger_3 = self.get_landmark(7, hand_landmarks, image)
                    index_finger_4 = self.get_landmark(8, hand_landmarks, image)
                    index_finger = [index_finger_1, index_finger_2, index_finger_3, index_finger_4]

                    if self.allnNone(*index_finger):
                        max_y = max(index_finger, key=lambda x: x[1])
                        min_y = min(index_finger, key=lambda x: x[1])

                        if max_y[1] - min_y[1] < HWPFD.FINGER_Y_THRESHOLD:
                            max_x = max(index_finger, key=lambda x: x[0])
                            direction = 'no direction'
                            if max_x == index_finger_1:
                                direction = 'left'
                            elif max_x == index_finger_4:
                                direction = 'right'
                            self.finger_direction_pub.publish(direction)
                            cv2.putText(image, f'Direction: {direction}', (10, H - ((1 * 20) + 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('Hands up & Finger Detection', image)
                if cv2.waitKey(1) & 0xFF == 27:
                    fps.stop()
                    print(fps.fps())
                    return

                fps.update()
                # self.rate.sleep()


if __name__ == '__main__':
    node = HWPFD(camera_topic='/image_raw/compressed')
    node.main()
