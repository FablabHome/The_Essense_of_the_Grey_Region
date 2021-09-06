#!/usr/bin/env python3

from core.Detection import PoseDetector, PoseRecognitionInput, PoseRecognitionProcess, PoseRecognitionDetector

from core.tools import OpenPose
import cv2 as cv
from tensorflow.keras.models import load_model
from os import path
import rospy

rospy.init_node('pose_recognition')

openpose_model_path = path.relpath(
    '../models/OpenPose/pose_iter_440000.caffemodel'
)

openpose_prototxt_path = path.realpath(
    '../../model_data/OpenPose/pose_deploy_linevec.prototxt'
)

pose = OpenPose(model_path=openpose_model_path, proto_path=openpose_prototxt_path)

# cap = Camera(require_cameras=(0, 1, 0, 0), rgb_topic='/bottom_camera/rgb/image_raw')
cap = cv.VideoCapture(0)

print('loading_model')
pose_recognize_model = load_model(path.relpath("../models/PoseDetection/model_0611.h5"))
print('model lodaded')
label = []
f = open(path.relpath("../../model_data/PoseDetection/model.txt"))
for line in f.readlines():
    label.append(line.strip())
f.close()

pose_detector = PoseDetector(pose)

pose_image_processor = PoseRecognitionInput(padding=(50, 70))
pose_output_processor = PoseRecognitionProcess()

pose_recognizer = PoseRecognitionDetector(
    detector=pose_recognize_model,
    image_processor=pose_image_processor,
    output_processor=pose_output_processor,
)

pose_box_color = {
    'stand': (0, 255, 32),
    'squat': (255, 32, 255),
    'sit': (255, 32, 255),
    'fall': (32, 0, 255),
    'not available': (255, 255, 255)
}

status_list = []
target = 'fall'

# pub = rospy.Publisher('/speaker_node/say', String, queue_size=1)
record_status_delay = rospy.get_rostime() + rospy.Duration(1)

while not rospy.is_shutdown():
    # srcframe = cap(image_id='rgb_compress')
    _, srcframe = cap.read()
    if srcframe is None:
        continue

    frame = srcframe.copy()
    image_shape = frame.shape

    pose_points = pose_detector.detect(frame)
    for pose_point in pose_points:
        if len(list(filter(lambda x: x[0] != -1, pose_point))) < 10:
            continue

        if pose_point is None:
            continue

        pose_box = pose_recognizer.detect(pose_point)
        if pose_box is None:
            continue

        confidence = pose_box.score
        status = pose_box.label

        color = pose_box_color[status]

        pose_box.draw(frame, color=color)
        pose_box.putText_at_top(frame, f'{status}: {confidence}', color=color)

        if rospy.get_rostime() >= record_status_delay:
            status_list.append(status)

        if len(status_list) > 20:
            status_list.pop(0)

        if status_list.count(target) > 10:
            status_list = list(filter(lambda x: x != target, status_list))
            # pub.publish(f'Somebody has fallen, please help')
            print('Somebody has fallen, please help')
            cv.putText(
                frame, 'Somebody has fallen, please help', (20, image_shape[0]), cv.FONT_HERSHEY_SIMPLEX, 1,
                (32, 0, 255), 2,
                cv.LINE_AA
            )

    frame = cv.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
    cv.imshow("pose", frame)
    key = cv.waitKey(1)

    if key in [ord('q'), 27]:
        break
