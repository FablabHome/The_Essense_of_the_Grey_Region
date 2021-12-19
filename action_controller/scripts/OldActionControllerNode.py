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
from os import path

import rospy
from home_robot_msgs.msg import CommandData
from mr_voice.msg import Voice
from std_msgs.msg import String

from core.Nodes import Node
from core.tools import Speaker
from core.utils.KeywordParsers import NormalKeywordParser


class ActionControllerNode(Node):
    def __init__(self):
        super(ActionControllerNode, self).__init__('acp', anonymous=False)

        config = rospy.get_param('~config')
        self.configs = json.load(open(config))
        self.action_commands = rospy.get_param('~action_commands')

        self.keyword_parser = NormalKeywordParser(self.configs)
        # self.acp_program = ActionController(node_id='acp', config_file=config)

        self.result = CommandData()

        # Buffer of requests, as queue
        self.buffer = []

        # Status of the ActionController
        self.is_running = False

        self.speaker = Speaker()

        # Create result publisher
        self.processed_result_publisher = rospy.Publisher(
            '~processed_result',
            CommandData,
            queue_size=1
        )

        # # #
        self.facial_pub = rospy.Publisher(
            '/home_edu/facial',
            String,
            queue_size=0
        )

        # Create Subscriber to subscribe for the text
        self.recognized_text_subscriber = rospy.Subscriber(
            '/voice/text',
            Voice,
            self._callback,
            queue_size=1
        )

        rospy.set_param('~is_running', self.is_running)

    def _callback(self, voice_data: Voice):
        text = voice_data.text
        # Append text into buffer
        self.buffer.append(text)
        # If Node is not running
        if not self.is_running:
            self.is_running = True
            # Pop buffer from queue
            while len(self.buffer) > 0:
                text = self.buffer.pop(0)
                # Get caller from _connection_header
                caller = voice_data._connection_header['callerid']
                rospy.loginfo(f'Processing request from {caller}')

                meaning, response, actions = self.keyword_parser.parse(text)
                self.speaker.say(response)

                if len(actions) != 0:
                    for action in actions:
                        self.__run_action(action)

                self.result.meaning = meaning
                self.result.response = response
                if not self.result.response != '':
                    self.facial_pub.publish(f"crying:Current text <{text}> doesn't match anybody in config")
                self.processed_result_publisher.publish(self.result)
            self.is_running = False

    def __run_action(self, action):
        # Get the command and args
        command, *args = action
        path_command = path.join(self.action_commands, command) + '.py'

        full_command = [path_command]
        full_command.extend(args)

        rospy.loginfo(f'Running action <{" ".join(action)}>')
        command_status = subprocess.run(full_command, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)

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
            rospy.logerr(
                f'Action <{" ".join(action)}> failed with exit code {command_status.returncode}')

        return command_status.returncode

    def main(self):
        pass

    def reset(self):
        self.result = CommandData()


if __name__ == '__main__':
    ac_node = ActionControllerNode()
    ac_node.spin()
