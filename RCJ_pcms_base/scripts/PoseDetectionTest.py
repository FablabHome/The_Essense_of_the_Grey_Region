#!/usr/bin/env python3

from os import path

import cv2 as cv
import imutils.video
import numpy as np
import rospy
from core.Detection import PoseDetector, PoseRecognitionInput, PoseRecognitionProcess, PoseRecognitionDetector
from core.tools import OpenPose, Camera
from std_msgs.msg import String
from tensorflow.keras.models import load_model


def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv.mean(roi)[:3]]
            cv.rectangle(image, (startX, startY), (endX, endY),
                         (B, G, R), -1)
    # return the pixelated blurred image
    return image


rospy.init_node('pose_recognition')

speaker_pub = rospy.Publisher('/speaker_node/say', String, queue_size=1)
# slam_pub = rospy.Publisher("/client/chassis_slam/point", Point, queue_size=1)
# cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=1)
#
# goal_point = Point(x=-0.65588299604, y=8.25853904999, z=0.994025686939)
# original_point = Point(x=2.08599934321, y=5.11935406091, z=0.703878971997)

base = path.realpath(__file__)
base = path.split(base)[0]
print(base)

openpose_model_path = path.join(
    base, '../models/OpenPose/pose_iter_440000.caffemodel'
)

openpose_prototxt_path = path.join(
    base, '../../model_data/OpenPose/pose_deploy_linevec.prototxt'
)

face_bin_path = path.join(
    base, '../models/intel/face-detection-retail-0004/FP16/face-detection-retail-0004.bin'
)

face_xml_path = path.join(
    base, '../models/intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml'
)

pose_recognizer = path.join(
    base, '../models/PoseDetection/model.h5'
)

pose = OpenPose(model_path=openpose_model_path, proto_path=openpose_prototxt_path)
face_detector = cv.dnn.readNet(face_bin_path, face_xml_path)

cap = Camera(require_cameras=(0, 1, 0, 0), rgb_topic='/top_camera/rgb/image_raw')

print('loading_model')
pose_recognize_model = load_model(pose_recognizer)
print('model lodaded')
label = []
f = open(path.join(base, '../../model_data/PoseDetection/model.txt'))
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
    'sit': (0, 255, 32),
    'squat': (0, 191, 255),
    'fall': (32, 0, 255)
}

status_list = []
target = 'fall'

record_status_delay = rospy.get_rostime() + rospy.Duration(1)

fps = imutils.video.FPS().start()

current_time = rospy.get_rostime()

while not rospy.is_shutdown():
    srcframe = cap(image_id='rgb_compress')
    if srcframe is None:
        continue

    frame = srcframe.copy()
    image_shape = frame.shape

    pose_points = pose_detector.detect(frame)

    face_blob = cv.dnn.blobFromImage(frame, size=(300, 300))
    face_detector.setInput(face_blob)
    faces = face_detector.forward()[0][0]

    for face in faces:
        _, _, conf, x1, y1, x2, y2 = face
        if conf == 0 or conf < 0.5882:
            continue

        x1, x2 = map(lambda x: int(x * image_shape[1]), [x1, x2])
        y1, y2 = map(lambda x: int(x * image_shape[0]), [y1, y2])



        face = frame[y1:y2, x1:x2, :].copy()
        blur = anonymize_face_pixelate(face)
        frame[y1:y2, x1:x2] = blur

    for pose_point in pose_points:
        pose.draw(frame, pose_point, thickness=5)

        pose_box = pose_recognizer.detect(pose_point)
        if pose_box is None:
            continue

        confidence = pose_box.score
        status = pose_box.label

        color = pose_box_color[status]

        pose_box.draw(frame, color=color)
        pose_box.putText_at_top(frame, f'{status}: {confidence:.2f}', color=color)

        if rospy.get_rostime() >= record_status_delay:
            status_list.append(status)

        if len(status_list) > 20:
            status_list.pop(0)

        if status_list.count(target) > 5:
            status_list = list(filter(lambda x: x != target, status_list))
            speaker_pub.publish(f'I saw somebody has fallen, please help')
            rospy.loginfo('Somebody has fallen, please help')

    frame = cv.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))
    cv.imshow("pose", frame)
    key = cv.waitKey(1)

    if key in [ord('q'), 27]:
        break

    fps.update()

fps.stop()
print(f'FPS: {fps.fps()}')
