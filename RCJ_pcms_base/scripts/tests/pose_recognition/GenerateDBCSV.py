#!/usr/bin/env python3
from core.tools import OpenPose

import logging
import progressbar
import cv2 as cv
import numpy as np
import pandas as pd
import os

base = './'

dataset_path = os.path.join(base, '../../../../datasets/dataset_pose')

pose_detector = OpenPose(
    os.path.join(base, '../../../models/OpenPose/pose_iter_440000.caffemodel'),
    os.path.join(base, '../../../../model_data/OpenPose/pose_deploy_linevec.prototxt')
)

csv_data = {}

bar = progressbar.ProgressBar(max_value=10)

logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='[%(levelname)s] - %(asctime)s - %(message)s',
    level=logging.DEBUG
)

for poses in os.listdir(dataset_path):
    if poses == 'sit':
        pose_data_path = os.path.join(dataset_path, poses)
        print(f'Processing class: {poses}')
        csv_path = os.path.join(pose_data_path, 'dataset.csv')

        rgb_path = os.path.join(pose_data_path, 'rgb')
        depth_path = os.path.join(pose_data_path, 'depth')

        count = 0
        for rgb, depth in zip(os.listdir(rgb_path), os.listdir(depth_path)):
            try:
                total = len(os.listdir(rgb_path))
                bar.max_value = total

                csv_data[rgb] = {}
                csv_data[rgb]['depth'] = depth
                csv_data[rgb]['points_data'] = []

                rgb_image = os.path.join(rgb_path, rgb)
                depth_image = os.path.join(depth_path, depth)

                rgb_image = cv.imread(rgb_image, cv.IMREAD_COLOR)
                depth_image = np.load(depth_image)['arr_0']

                if poses == 'fall':
                    rgb_image[156:211, 268:307] = 0
                if poses == 'sit':
                    rgb_image[188:323, 123:172] = 0

                points = pose_detector.detect(rgb_image)
                for point in points:
                    pose_detector.draw(rgb_image, point)

                cv.imshow('frame', rgb_image)
                cv.waitKey(16)

                if len(points) > 1 or len(points) == 0:
                    csv_data.pop(rgb, None)
                    continue

                for x, y in points[0]:
                    distance = depth_image[y, x]
                    if distance == 0:
                        for each in range(1, depth_image.shape[0]):
                            up = y - each
                            down = y + each
                            left_x = x - each
                            right_x = x + each

                            top = depth_image[up:up + 1, left_x:right_x + 1]
                            left = depth_image[up:up + 1, left_x:left_x + 1]
                            bottom = depth_image[down:down + 1, left_x:left_x + 1]
                            right = depth_image[up:down + 1, right_x:right_x + 1]

                            for block in [top, left, bottom, right]:
                                nonzero = block[np.nonzero(block)]
                                if nonzero.shape[0] > 0:
                                    distance = nonzero[0]
                                    break

                    csv_data[rgb]['points_data'].append([x, y, distance])

                csv_data[rgb]['label'] = poses
                logging.info(f'{rgb}: Successfully processed')
            except Exception as e:
                logging.error(f'{rgb}: Skipped image due to error, msg: {e}')
                continue

            bar.update(count)
            count += 1

        df = pd.DataFrame(data=csv_data).T
        df.to_csv(csv_path)
