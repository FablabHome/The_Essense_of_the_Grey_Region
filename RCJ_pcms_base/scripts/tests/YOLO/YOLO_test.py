import os

import cv2 as cv
from PIL import Image
from rospkg import RosPack

from keras_yolo3.yolo import YOLO_np

base = RosPack().get_path('rcj_pcms_base') + '/..'
_model_h5 = os.path.join(base, 'models/YOLO/yolov3.h5')
_coco_classes = os.path.join(base, 'models/YOLO/coco_classes.txt')
_anchors = os.path.join(base, 'models/YOLO/yolo3_anchors.txt')
_model = YOLO_np(
    model_type='yolo3_darknet',
    yolo_weights_path='',
    weights_path=_model_h5,
    anchors_path=_anchors,
    classes_path=_coco_classes
)

coco_classes = open(_coco_classes).readlines()

cap = cv.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    if frame is None:
        continue

    H, W, _ = frame.shape

    blob = Image.fromarray(frame)
    _, out_boxes, out_classes, _ = _model.detect_image(blob)
    for box, label in zip(out_boxes, out_classes):
        x1, y1, x2, y2 = box
        cv.rectangle(frame, (x1, y1), (x2, y2), (32, 255, 0), 3)

    # _net.setInput(blob)
    # dets = _net.forward()

    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key in [ord('q'), 27]:
        break

cap.release()
cv.destroyAllWindows()
