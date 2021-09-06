from typing import List
import warnings
try:
    import dlib
except ImportError:
    warnings.warn('Dlib rectangle feature will be disabled', ImportWarning)
import numpy as np
from home_robot_msgs.msg import ObjectBox
from imutils.object_detection import non_max_suppression

from core.Dtypes import BBox

get_pos_from_object_box = np.vectorize(lambda box: (box.x1, box.y1, box.x2, box.y2, box.label, box.score, box.model))


def filterBoxes(out_boxes, min_area):
    out_boxes = filter(lambda x: x.area >= min_area, out_boxes)
    return list(out_boxes)


def posToBBox(out_boxes, padding=None, shape=None):
    for box_data in out_boxes:
        x1, y1, x2, y2, label, score, model = box_data
        yield BBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            label=label,
            score=score,
            padding=padding,
            shape=shape,
            model=model
        )


def dlibToBBox(out_boxes, padding=None, shape=None):
    for dlib_box_data in out_boxes:
        dlib_box, label, score = dlib_box_data
        yield BBox(
            x1=dlib_box.left(),
            y1=dlib_box.top(),
            x2=dlib_box.right(),
            y2=dlib_box.bottom(),
            label=label,
            score=score,
            padding=padding, shape=shape
        )


def deserialize_ros_to_bbox(object_boxes: List[ObjectBox]):
    if len(object_boxes) == 0:
        return []

    boxes_data = get_pos_from_object_box(object_boxes)
    boxes_data = np.array(boxes_data)
    boxes_data[:, :4] = boxes_data[:, :4].astype(int)

    return posToBBox(out_boxes=boxes_data)


def BBoxToPos(out_boxes):
    for box in out_boxes:
        yield box.as_np_array()


def do_nms(boxes, use_bbox=True, padding=None, shape=None):
    processed_box = non_max_suppression(boxes)

    if use_bbox:
        return posToBBox(processed_box, padding, shape)

    return processed_box
