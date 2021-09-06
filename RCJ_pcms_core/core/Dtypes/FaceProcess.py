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
import pickle
import os
import pprint
from typing import List

import numpy as np
import pandas as pd


class FaceUser:
    def __init__(self, username: str, description: np.array):
        self.username = username
        self.description = description

    def __repr__(self):
        return f'username:{self.username}'

    def compare_with_another_face(self, another_face):
        return np.sqrt(np.sum(np.square(self.description - another_face.description)))


class FaceUserManager:
    def __init__(self, database: str):
        self.database: str = database
        self.users: List[FaceUser] = []

        # Load database base on format of database
        if database.endswith('pickle'):
            self._load_pickle_database(database)
        elif os.path.isdir(database):
            self._load_directory_csv_database(database)

    def __repr__(self):
        return f'FaceUserManager at {hex(id(self))} with users:\n{pprint.pformat(self.users)}'

    def _load_pickle_database(self, database):
        # Read the pickle database file
        unconverted_users = pickle.load(open(database, 'rb'))
        # Pickle format:
        # {<username1>: <description1>, <username2>: <description2>}
        for username, description in unconverted_users.items():
            # Get users from pickle and convert to FaceUser
            self.users.append(FaceUser(username=username, description=np.array(description)))

    def _load_directory_csv_database(self, database):
        for user_csv in os.listdir(database):
            # Read csv userdata using pandas
            csv_userdata: pd.DataFrame = pd.read_csv(f'{database}/{user_csv}.csv')

            # CSV format:
            # Column 1: <username>
            # Data of Column 1: <description>
            username = csv_userdata.columns[0]
            description = np.array(csv_userdata[username])

            # Convert data into FaceUser
            self.users.append(FaceUser(username=username, description=description))

    def sign_in(self, face: FaceUser) -> FaceUser:
        for seen_user in self.users:
            if face.compare_with_another_face(seen_user) < 0.4:
                return seen_user
        else:
            return FaceUser(username='0', description=[])
