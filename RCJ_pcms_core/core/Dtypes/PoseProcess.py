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
z6
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from ..utils.boxing import posToBBox

import cv2 as cv
import numpy as np


class PoseGesture:
    def __init__(self, pose_points: np.array):
        # Initialize pose gesture points
        self.one_person_gesture_points = pose_points

        # Skeleton point pairs
        self.pairs = (
            (1, 0),
            (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13),
            (0, 14), (14, 16),
            (0, 15), (15, 17)
        )
        # Skeleton colors
        self.colors = (
            (1, 255, 1),
            (1, 255, 255), (128, 128, 255), (196, 196, 1),
            (255, 1, 255), (255, 128, 128), (1, 196, 196),
            (1, 1, 255), (255, 255, 1), (1, 140, 140),
            (255, 1, 1), (1, 128, 255), (140, 1, 140),
            (196, 1, 196), (128, 1, 128),
            (1, 196, 1), (1, 128, 1)
        )
        
    def draw(self, image, pose_points=None, thickness=2):
        if pose_points is None:
            pose_points = self.one_person_gesture_points

        for i, pair in enumerate(self.pairs):
            x1 = pose_points[pair[0]][0]
            y1 = pose_points[pair[0]][1]
            x2 = pose_points[pair[1]][0]
            y2 = pose_points[pair[1]][1]
            if x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1:
                continue
            cv.line(image, (x1, y1), (x2, y2), self.colors[i], thickness)

    def to_box(self, to_bbox=False):
        gesture_x = self.one_person_gesture_points[:, 0]
        gesture_y = self.one_person_gesture_points[:, 1]

        # Filter values smaller than zero
        masked_gesture_x = gesture_x[np.where(gesture_x > 0)]
        masked_gesture_y = gesture_y[np.where(gesture_y > 0)]

        if len(masked_gesture_x) == 0 or len(masked_gesture_y) == 0:
            return None

        # Get box coordinates
        x1 = min(masked_gesture_x)
        y1 = min(masked_gesture_y)
        x2 = max(masked_gesture_x)
        y2 = max(masked_gesture_y)

        if to_bbox:
            return list(posToBBox([[x1, y1, x2, y2, '', 1., 'openpose']]))[0]

        return x1, y1, x2, y2

    def to_black_board(self):
        new_data = []
        try:
            # Get box of pose_gesture
            x1, y1, x2, y2 = self.to_box()
        except:
            return None

        for j in range(18):
            num_x = self.one_person_gesture_points[j][0]
            num_y = self.one_person_gesture_points[j][1]

            if num_x and num_y > 0:
                num_x = num_x - x1
                num_y = num_y - y1
                new_data.append((num_x, num_y))
            else:
                num_x = -1
                num_y = -1
                new_data.append((num_x, num_y))

        height = y2 - y1
        width = x2 - x1

        image = np.zeros((height, width, 3), np.uint8)

        new_image = np.zeros((100, 100, 3), np.uint8)
        self.draw(image, new_data)

        n_w = 100
        n_h = 100

        if image.shape[1] > image.shape[0]:
            n_h = int(image.shape[0] * n_w / image.shape[1])
        else:
            if image.shape[0] == 0:
                return None

            n_w = int(image.shape[1] * n_h / image.shape[0])

        if n_w == 0 or n_h == 0:
            return None

        ox = int((100 - n_w) / 2)
        oy = int((100 - n_h) / 2)

        # print(oy, oy+n_h, ox, ox+n_w)

        image = cv.resize(image, (n_w, n_h))
        new_image[oy:n_h + oy, ox:n_w + ox, :] = image[0:image.shape[0], 0:image.shape[1], :]

        return new_image
