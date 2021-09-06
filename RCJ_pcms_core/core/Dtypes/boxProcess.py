#!/usr/bin/env python3

"""
MIT License

Copyright (c) 2019 rootadminWalker

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

from math import sqrt
import warnings
import cv2 as cv
try:
    import dlib
except ImportError:
    warnings.warn('Dlib rectangle feature will be disabled', ImportWarning)

import numpy as np
from home_robot_msgs.msg import ObjectBox


class BBox:
    """
    BBox is a helper type for some convenience methods and calculations
    1.1 Call
    method 1
    box = BBox({
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    }, padding=(top, side), shape=image_shape)

    padding (Optional, tuple): This create a same ratio but bigger box.
    shape (Optional, tuple): Use with padding to avoid override

    method 2
    box = posToBBox([[x1, y1, x2, y2], [x1, y1, x2, y2]], padding=(top, side), shape=image_shape)

    1.2 Convenience variables

    Get centroid by
    --box.centroid

    Area by
    --box.area

    x1, y1, x2, y2 by
    --box.x1, box.y1, box.x2, box.y2

    Padding-box by
    --box.padding_box

    Height and width by
    --box.height, box.width

    1.3 Convenience methods
    --box.is_inBox(box2)

    Desc:
    Check if box is in box2

    Args:
    box2 (Required, BBox)

    --box.calc_distance_between_boxes(box2)

    Desc:
    Check distance between self and box2

    Args:
    box2 (Required, BBox)

    --box.draw(image, color, thickness)

    Desc:
    Draw itself to a OpenCV image

    Args:
    image (Required, np.array): A OpenCV image
    color (Required, tuple): The color of the rectangle
    thickness (Required, tuple): Thickness of rectangle

    --box.draw_centroid(image, color, thickness)

    Desc:
    Draws itself's centroid to an OpenCV image

    Args:
    image (Required, np.array): A OpenCV image
    color (Required, tuple): The color of the centroid
    thickness (Required, tuple): Thickness of centroid

    --box.as_np_array()

    Desc:
    Return itself as np array coordinates. format: [x1, y1, x2, y2]

    """

    def __init__(
            self,
            x1: int = 0,
            y1: int = 0,
            x2: int = 0,
            y2: int = 0,
            coordinates: dict = None,
            label: str = '',
            model: str = '',
            score: float = 0.0,
            source_img=None,
            padding=None,
            shape=None
    ):
        self.label = label
        self.coordinates = coordinates
        self.padding = padding
        self.score = score
        self.model = model

        if self.coordinates is not None:
            for key, value in self.coordinates.items():
                self.coordinates[key] = int(value)

            self.x1 = self.coordinates['x1']
            self.y1 = self.coordinates['y1']
            self.x2 = self.coordinates['x2']
            self.y2 = self.coordinates['y2']
        else:
            self.x1 = x1
            self.y1 = y1
            self.x2 = x2
            self.y2 = y2

        if shape is not None and padding is not None:
            self.padding_box = self.__calc_padding(shape)

        self.area = self.__calc_area()
        self.centroid = self.__calc_centroid()

        self.corresponds = {
            'x1': ['width', 'x2'],
            'y1': ['height', 'y2'],
            'x2': ['width', 'x1'],
            'y2': ['height', 'y1'],
        }

        self.height = self.y2 - self.y1
        self.width = self.x2 - self.x1

        self.source_img = source_img

        self.serialize_msg = ObjectBox()

    def __repr__(self):
        return '{} at {}, {}, pos=(x1={}, y1={}, x2={}, y2={}), centroid={}, area={}>'.format(
            self.__class__, hex(id(self)),
            f'label: {self.label}' if self.label is not '' else 'no label',
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            self.centroid,
            self.area
        )

    def __calc_centroid(self):
        return int(((self.x2 - self.x1) / 2) + self.x1), \
               int(((self.y2 - self.y1) / 2) + self.y1)

    def __calc_area(self) -> int:
        self.area = (self.y2 - self.y1) * (self.x2 - self.x1)
        return self.area

    def __calc_padding(self, shape):
        px1 = self.x1 - self.padding[0]
        py1 = self.y1 - self.padding[1]
        px2 = self.x2 + self.padding[0]
        py2 = self.y2 + self.padding[1]

        self.padding_box = {
            'x1': px1 if px1 > 0 else 0,
            'y1': py1 if py1 > 0 else 0,
            'x2': px2 if px2 < shape[1] else shape[1],
            'y2': py2 if py2 < shape[0] else shape[0]
        }
        return BBox(coordinates=self.padding_box)

    def is_inBox(self, box2):
        isX1 = self.x1 - box2.x1 >= 0
        isY1 = self.y1 - box2.y1 >= 0
        isX2 = self.x2 - box2.x2 <= 0
        isY2 = self.y2 - box2.y2 <= 0

        return isX1 and isY1 and isX2 and isY2

    def calc_distance_between_point(self, point2):
        x_distance = self.centroid[0] - point2[0]
        y_distance = self.centroid[1] - point2[1]
        return sqrt((x_distance ** 2) + (y_distance ** 2))

    def draw(self, image, color=(255, 32, 255), thickness=5):
        cv.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), color, thickness)

    def draw_centroid(self, image, color, radius=5, thickness=-1):
        cv.circle(image, self.centroid, radius, color, thickness)

    def putText_at_top(self, image, text, color=(255, 32, 255), thickness=2, font_scale=1):
        text_origin = (self.x1, int(self.y1 - thickness - font_scale))
        cv.rectangle(image, (self.x1, text_origin[1] - 30), (self.x2, self.y1), color, -1)
        cv.putText(
            image,
            text,
            text_origin,
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale, (0, 0, 0), thickness,
            cv.LINE_AA
        )

    def bb_iou_score(self, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.x1, boxB.x1)
        yA = max(self.y1, boxB.y1)
        xB = min(self.x2, boxB.x2)
        yB = min(self.y2, boxB.y2)
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = self.area
        boxBArea = boxB.area
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def as_np_array(self) -> np.array:
        return np.array([self.x1, self.y1, self.x2, self.y2])

    # def as_dlib_rectangle(self) -> dlib.rectangle:
    #     return dlib.rectangle(self.x1, self.y1, self.x2, self.y2)

    def as_list(self) -> list:
        return [self.x1, self.y1, self.x2, self.y2]

    def divide_part_of_box(self, option, value):
        width_or_height = self.corresponds[option][0]
        correspond_pos = self.corresponds[option][1]

        self.__dict__[width_or_height] *= value
        result_pos = int(self.__dict__[width_or_height] - self.__dict__[correspond_pos])

        self.__dict__[option] = result_pos if result_pos > 0 else -result_pos

    def crop_box_from_image(self, image):
        image = image[self.y1:self.y2, self.x1:self.x2, :].copy()
        return image

    def serialize_ros(self):
        self.serialize_msg.x1 = self.x1
        self.serialize_msg.y1 = self.y1
        self.serialize_msg.x2 = self.x2
        self.serialize_msg.y2 = self.y2

        self.serialize_msg.model = self.model
        self.serialize_msg.score = self.score
        self.serialize_msg.label = self.label

        return self.serialize_msg
