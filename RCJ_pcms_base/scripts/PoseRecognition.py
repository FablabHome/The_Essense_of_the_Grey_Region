#!/usr/bin/env python3

from os import path

import cv2 as cv
import numpy as np
import rospy
import tensorflow as tf
from cv_bridge import CvBridge
from robot_vision_msgs.msg import HumanPoses
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage
from tensorflow.keras.models import load_model

from core.Dtypes import PoseGesture
from core.Nodes import Node

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
cfgs = tf.compat.v1.ConfigProto(gpu_options=opts)
sess = tf.compat.v1.Session(config=cfgs)


class PoseRecognition(Node):
    def __init__(self):
        super(PoseRecognition, self).__init__('pose_recognition')
        base = RosPack().get_path('rcj_pcms_base') + '/..'
        pose_recognizer_path = path.join(base, 'models/PoseDetection/model_6.h5')
        self.pose_recognizer = load_model(pose_recognizer_path)

        self.bridge = CvBridge()

        self.stat_pub = rospy.Publisher(
            '~stats',
            HumanPoses,
            queue_size=1
        )

        rospy.Subscriber(
            '/openpose_ros/human_poses',
            HumanPoses,
            self.pose_callback,
            queue_size=1
        )
        rospy.Subscriber(
            '/bottom_camera_rotate/rgb/image_raw/compressed',
            CompressedImage,
            self.camera_callback,
            queue_size=1
        )

        self.src_poses = []
        self.srcframe = None
        self.labels = ['danger', 'safe', 'warn', 'safe']
        self.main()

    def pose_callback(self, poses: HumanPoses):
        self.src_poses = poses.poses

    def camera_callback(self, image: CompressedImage):
        self.srcframe = self.bridge.compressed_imgmsg_to_cv2(image)

    def main(self):
        while not rospy.is_shutdown():
            if self.srcframe is None:
                continue

            frame = self.srcframe.copy()
            poses = self.src_poses.copy()

            for pose in poses:
                apose = np.array([
                    (pose.Nose.x, pose.Nose.y),
                    (pose.Chest.x, pose.Chest.y),
                    (pose.RShoulder.x, pose.RShoulder.y),
                    (pose.RElbow.x, pose.RElbow.y),
                    (pose.RWrist.x, pose.RWrist.y),
                    (pose.LShoulder.x, pose.LShoulder.y),
                    (pose.LElbow.x, pose.LElbow.y),
                    (pose.LWrist.x, pose.LWrist.y),
                    (pose.RHip.x, pose.RHip.y),
                    (pose.RKnee.x, pose.RKnee.y),
                    (pose.RAnkle.x, pose.RAnkle.y),
                    (pose.LHip.x, pose.LHip.y),
                    (pose.LKnee.x, pose.LKnee.y),
                    (pose.LAnkle.x, pose.LAnkle.y),
                    (pose.REye.x, pose.REye.y),
                    (pose.LEye.x, pose.LEye.y),
                    (pose.REar.x, pose.REar.y),
                    (pose.LEar.x, pose.LEar.y)
                ])
                pose_gesture = PoseGesture(apose)
                black_board = pose_gesture.to_black_board()
                if black_board is None:
                    continue

                preds = self.pose_recognizer.predict(
                    np.array([black_board]))[0]

                h_rate = np.argmax(preds)
                status = self.labels[h_rate]
                pose.pose = status

                pose_gesture.draw(frame, thickness=5)
                box = pose_gesture.to_box(to_bbox=True)

                color = (32, 255, 0)
                if status == 'danger':
                    color = (32, 0, 255)
                elif status == 'warn':
                    color = (0, 255, 255)

                box.draw(frame, color=color)
                box.putText_at_top(frame, f'{status}', color, 2, 0.9)

            msg = HumanPoses()
            msg.poses = poses
            self.stat_pub.publish(msg)
            cv.imshow('frame', frame)
            key = cv.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

    def reset(self):
        pass


if __name__ == '__main__':
    node = PoseRecognition()
