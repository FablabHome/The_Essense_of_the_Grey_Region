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

import rospy
from home_robot_msgs.msg import CommandData
from mr_voice.msg import Voice
from std_msgs.msg import String

from core.Nodes.ActionController import ActionController
from core.Nodes import Node


class ActionControllerNode(Node):
    def __init__(self):
        super(ActionControllerNode, self).__init__('acp', anonymous=False)
        self.acp_program = ActionController(node_id='acp', config_file='./config/wrs_demo_2.json')

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

        self.speaker_pub = rospy.Publisher(
            '/speaker/say',
            String,
            queue_size=1
        )

        # Create Subscriber to subscribe for the text
        self.recognized_text_subscriber = rospy.Subscriber(
            '/voice/text',
            Voice,
            self._callback,
            queue_size=1
        )

        self.result = CommandData()

        # Buffer of requests, as queue
        self.buffer = []

        # Status of the ActionController
        self.is_running = False
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

                # Run acp process and publish the result
                self.result = self.acp_program.run(text, serialize=True)
                if not self.result.response != '':
                    self.facial_pub.publish(f"crying:Current text <{text}> doesn't match anybody in config")
                self.processed_result_publisher.publish(self.result)
            self.is_running = False

    def main(self):
        pass

    def reset(self):
        self.result = CommandData()


if __name__ == '__main__':
    ac_node = ActionControllerNode()
    ac_node.spin()
