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
import json
import subprocess
from os.path import join
from typing import Any

import rospy
from home_robot_msgs.msg import CommandData
from rospkg import RosPack
from std_msgs.msg import String

from ..base_classes import NodeProgram


class ActionController(NodeProgram):
    BASE = RosPack().get_path('rcj_pcms_base')

    def __init__(
            self,
            node_id,
            config_file: str,
            action_commands: str = join(BASE, './scripts/action_commands')
    ):
        super(ActionController, self).__init__(node_id)
        self.speaker_pub = rospy.Publisher(
            '/speaker/say',
            String,
            queue_size=1
        )
        rospy.sleep(.5)

        self.config_file = config_file
        self.action_commands = action_commands

        self.configs = json.load(open(join(ActionController.BASE, self.config_file), 'r'))

        self.require_keywords_status = False
        self.separately_keywords_status = False

        self.require_keywords = []
        self.separately_keywords = []

        self.serialize_msg = CommandData()

        self.meaning = self.response = ''
        self.actions = []

    def run(self, text: str, serialize: bool = False) -> (str, str, list):
        for meaning, configs in self.configs.items():
            self.require_keywords = configs['and']
            self.separately_keywords = configs['or']
            rospy.loginfo(f'{self.require_keywords}, {self.separately_keywords}')

            self._has_require_keywords(text)
            self._has_separately_keywords(text)

            if self.require_keywords_status and self.separately_keywords_status:
                # Get response and action
                self.response = configs['response']
                self.actions = configs['action']
                self.meaning = meaning

                self.speaker_pub.publish(String(self.response))

                rospy.loginfo(f'Text \'{text}\' matched, Meaning: {meaning}, response: {self.response}')

                if len(self.actions) != 0:
                    for action in self.actions:
                        # Get the command and args
                        command, *args = action
                        path_command = join(self.action_commands, command) + '.py'

                        full_command = [path_command]
                        full_command.extend(args)

                        command_status = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        # Decode the stdout
                        stdout = command_status.stdout.decode('ascii')
                        # Check the returncode
                        if command_status.returncode == 0:
                            # Return code 0 means program executed successfully
                            rospy.loginfo(f'Action <{" ".join(action)}> has run successfully')
                            if stdout != '':
                                rospy.loginfo(f'stdout: {stdout}')
                        else:
                            # If status code is not 0, which means action stopped unexpectedly
                            # Log the error
                            rospy.logerr(f'Action <{" ".join(action)}> failed with exit code {command_status.returncode}')
                break
        else:
            rospy.logerr(f'Text \'{text}\' doesn\'t match anybody in the config file')
            self.response = self.meaning = ''

        if serialize:
            return self.serialize_output()
        return self.meaning, self.response, self.actions

    @staticmethod
    def _input_text_processor(text):
        return text.strip().lower()

    def _has_require_keywords(self, text):
        input_text = self._input_text_processor(text)
        if len(self.require_keywords) == 0:
            self.require_keywords_status = True
            return

        for require_keyword in self.require_keywords:
            require_keyword = self._input_text_processor(require_keyword)
            if require_keyword not in input_text:
                self.require_keywords_status = False
                break
        else:
            self.require_keywords_status = True

    def _has_separately_keywords(self, text):
        input_text = self._input_text_processor(text)
        if len(self.separately_keywords) == 0:
            self.separately_keywords_status = True
            return

        for separately_keyword in self.separately_keywords:
            separately_keyword = self._input_text_processor(separately_keyword)
            if separately_keyword in input_text:
                self.separately_keywords_status = True
                break
        else:
            self.separately_keywords_status = False

    def serialize_output(self) -> Any:
        self.serialize_msg.meaning = self.meaning
        self.serialize_msg.response = self.response
        # self.serialize_msg.action = ';'.join(self.actions)
        return self.serialize_msg
