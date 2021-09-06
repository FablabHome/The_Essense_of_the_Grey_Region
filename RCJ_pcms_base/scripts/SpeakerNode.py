#!/usr/bin/env python3
import threading

import pyttsx3
import rospy
from std_msgs.msg import String

from core.base_classes import Node


class SpeakerNode(Node):
    def __init__(
        self,
        name: str = 'node',
        anonymous: bool = False,
        language: str = 'en-us',
        rate: int = 130
    ):
        super(SpeakerNode, self).__init__(name, anonymous)
        self.engine = pyttsx3.init()
        self.engine.setProperty('voice', language)
        self.engine.setProperty('rate', rate)

        self.engine.connect('started-utterance', self.on_start)
        self.engine.connect('started-word', self.on_word)
        self.engine.connect('finished-utterance', self.on_end)

        self.is_running = False
        self.buffer = []

        self.lock = threading.RLock()

        rospy.Subscriber(
            '~text',
            String,
            self.callback,
            queue_size=1
        )

    def on_start(self, name):
        self.is_running = True

    def on_word(self, name, location, length):
        pass

    def on_end(self, name, completed):
        self.is_running = False

    def callback(self, message: String):
        self.buffer.append(message.data)
        if not self.is_running:
            while len(self.buffer) > 0:
                rospy.loginfo(message.data)
                self.engine.say(self.buffer.pop(0))
                self.engine.runAndWait()
                self.engine.stop()

    def reset(self):
        self.engine = pyttsx3.init()


if __name__ == '__main__':
    node = SpeakerNode('speaker', language='zh', rate=250)
    node.spin()
