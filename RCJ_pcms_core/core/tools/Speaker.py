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
from mr_voice.srv import SpeakerSrv
from std_msgs.msg import String

from .Abstract import Tools


class Speaker(Tools):
    def __init__(self, speaker_topic='/speaker/say', speaker_srv='/speaker/text'):
        super()._check_status()
        self.speaker_srv = rospy.ServiceProxy(speaker_srv, SpeakerSrv)
        self.speaker_pub = rospy.Publisher(
            speaker_topic,
            String,
            queue_size=1
        )

    def say(self, text, wait_until_end=False):
        if wait_until_end:
            self.speaker_srv(text)
        else:
            self.speaker_pub.publish(text)
