import json
import logging
import os
import time

import warnings

import cv2 as cv
import numpy as np
import progressbar

from core.tools import OpenPose
from core.Dtypes import PoseGesture

from os import path
import pandas as pd

# Get base directory
base = path.split(path.realpath(__file__))[0]

# Initialize OpenPose detector
# TODO: Pack detecting steps into classes
pose_detector = OpenPose(
    path.join(base, '../../../models/OpenPose/pose_iter_440000.caffemodel'),
    path.join(base, '../../../../model_data/OpenPose/pose_deploy_linevec.prototxt')
)

logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

# Database path
database = '/home/root_walker/workspace/datasets/dataset_pose_test/'

# Dataframe CSV path
csv_path = '/home/root_walker/workspace/datasets/dataset_pose_test/dataset.csv'
df = pd.read_csv(csv_path)

# Getting line data
rgbs = df['rgb']
depths = df['depth']
points_data = df['points_data']
labels = df['label']

# Create a progressbar for knowing the progress
bar = progressbar.ProgressBar(max_value=len(rgbs))

count = 0

# TODO: Find a way to process the depth points data
for rgb, depth, one_person_points_data, label in zip(rgbs, depths, points_data, labels):
    try:
        onlypose_path = path.join(database, label, 'onlypose')
        if not path.exists(onlypose_path): os.mkdir(onlypose_path)

        # Get rgb and depth image paths
        rgb_img_name = path.join(database, label, 'rgb', rgb)
        depth_img_name = path.join(database, label, 'depth', depth)

        # Load depth and rgb images
        rgb_image = cv.imread(rgb_img_name)
        depth_image = np.load(depth_img_name)['arr_0']

        one_person_gesture = json.loads(one_person_points_data)
        one_person_gesture = PoseGesture(np.array(one_person_gesture))

        one_person_gesture.draw(rgb_image)
        black_board = one_person_gesture.to_black_board()

        cv.imwrite(path.join(onlypose_path, f'{time.time()}'.replace('.', '') + '.jpg'), black_board)

        cv.imshow('frame', rgb_image)
        cv.imshow('black_board', black_board)
        cv.waitKey(16)

        logging.info(f'{rgb}, Created successfully')

    except Exception as e:
        logging.error(f'{rgb}, Skipped image due to error, msg: {e}')
        warnings.warn(f'Skipped image due to error, msg: {e}', RuntimeWarning)

    bar.update(count)
    count += 1
