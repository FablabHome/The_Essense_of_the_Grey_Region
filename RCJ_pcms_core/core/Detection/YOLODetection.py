#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2020 rootadminWalker

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
import cv2
import numpy as np


class DetectBox(object):
    LABELS = (
        "person", "bicycle", "car", "motorbike", "aeroplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "sofa", "potted plant", "bed",
        "dining table", "toilet", "tv monitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )
    SIDE = 13
    ANCHORS = {
        13: ((116, 90), (156, 198), (373, 326)),
        26: ((30, 61), (62, 45), (59, 119)),
        52: ((10, 13), (16, 30), (33, 23))
    }
    COORDINATE = 4
    THRESHOLD = 0.7
    IMG_W, IMG_H = 416, 416

    def __init__(self, x, y, w, h, class_id, confidence, img_w, img_h):
        w_scale = img_w / self.IMG_W
        h_scale = img_h / self.IMG_H
        self.x_min = max(0, int((x - w / 2) * w_scale))
        self.y_min = max(0, int((y - h / 2) * h_scale))
        self.x_max = min(img_w, int((x + w / 2) * w_scale))
        self.y_max = min(img_h, int((y + h / 2) * h_scale))
        self.class_id = class_id
        self.confidence = confidence

    def area(self) -> float:
        return (self.y_max - self.y_min) * (self.x_max - self.x_min)

    def draw(self, image, color=(64, 128, 64), thick=2, f_color=(255, 255, 255), f_scale=0.5, f_thick=1):
        cv2.rectangle(
            image,
            (self.x_min, self.y_min),
            (self.x_max, self.y_max),
            color, thick
        )
        text = "%s %.2f" % (self.LABELS[self.class_id], self.confidence)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        w, h = cv2.getTextSize(text, font, f_scale, f_thick)[0]
        cv2.rectangle(
            image,
            (self.x_min, self.y_min - h),
            (self.x_min + w, self.y_min),
            color, -1
        )
        cv2.putText(
            image, text, (self.x_min, self.y_min),
            font, scale, f_color, f_thick, cv2.LINE_AA
        )

    @classmethod
    def calc_intersection_over_union(cls, box1, box2) -> float:
        w_overlap_area = min(box1.x_max, box2.x_max) - max(box1.x_min, box2.x_min)
        h_overlap_area = min(box1.y_max, box2.y_max) - max(box1.y_min, box2.y_min)
        overlap_area = 0.0
        if w_overlap_area > 0.0 and h_overlap_area > 0.0:
            overlap_area = w_overlap_area * h_overlap_area
        union_area = box1.area() + box2.area() - overlap_area
        if union_area <= 0.0:
            return 0.0
        return overlap_area / union_area

    @classmethod
    def parse_output(cls, output: np.ndarray, img_w: int, img_h: int) -> []:
        ret = []
        for row in range(cls.SIDE):
            for col in range(cls.SIDE):
                for n in range(len(cls.ANCHORS[cls.SIDE])):
                    index = n * (len(cls.LABELS) + cls.COORDINATE + 1)
                    confidence = output[0][index + cls.COORDINATE][row][col]
                    if confidence < cls.THRESHOLD:
                        continue
                    max_prob = 0.0
                    max_bbox = None
                    for i in range(len(cls.LABELS)):
                        class_index = index + cls.COORDINATE + 1 + i
                        prob = output[0][class_index][row][col] * confidence
                        if prob < cls.THRESHOLD or prob < max_prob:
                            continue
                        x = (col + output[0][index + 0][row][col]) * cls.IMG_W / cls.SIDE
                        y = (row + output[0][index + 1][row][col]) * cls.IMG_H / cls.SIDE
                        w = np.exp(output[0][index + 2][row][col]) * cls.ANCHORS[cls.SIDE][n][0]
                        h = np.exp(output[0][index + 3][row][col]) * cls.ANCHORS[cls.SIDE][n][1]
                        max_prob = prob
                        max_bbox = DetectBox(x, y, w, h, i, prob, img_w, img_h)
                    if max_bbox is not None:
                        ret.append(max_bbox)

        for i in range(len(ret)):
            for j in range(i + 1, len(ret)):
                if cls.calc_intersection_over_union(ret[i], ret[j]) >= 0.4:
                    ret[j].confidence = 0.0

        for i in range(len(ret) - 1, -1, -1):
            if ret[i].confidence < 0.2:
                ret.pop(i)

        return ret
