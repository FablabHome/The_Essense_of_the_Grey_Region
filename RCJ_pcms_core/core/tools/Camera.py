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

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import rospy


class Camera:
    def __init__(
            self,
            require_cameras: (bool, bool, bool, bool) = (True, False, False, False),
            rgb_topic: str = '/camera/rgb/image_raw',
            depth_topic: str = '/camera/depth/image_raw',
            compress_topic: str = '/compressed'
    ):
        """
        Args:
            require_cameras: Chose camera to open, corresponding: (rgb_camera, rgb_compress, depth_camera,
                depth_compress)
            rgb_topic: Topic for rgb camera
            depth_topic: Topic for depth camera
            compress_topic: Compress image topic for both cameras
        """

        self.require_cameras = require_cameras
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.compress_topic = compress_topic

        self.rgb_compress_topic = self.rgb_topic + self.compress_topic
        self.depth_compress_topic = self.depth_topic + self.compress_topic

        self.rgb_image = self.rgb_compress_image = self.depth_image = self.depth_compress_image = None

        self.require_cameras_iterator = iter(self.require_cameras)

        self.bridge = CvBridge()

        if rospy.get_name() == '/unnamed':
            raise rospy.ROSException('Node not register, call rospy.init_node first')

        self.image_ids = {
            'rgb': self.rgb_image,
            'rgb_compress': self.rgb_compress_image,
            'depth': self.depth_image,
            'depth_compress': self.depth_compress_image
        }

        try:
            self.cameras = {
                self.rgb_topic: {
                    'data_class': Image,
                    'callback': self._rgb_callback,
                    'open': next(self.require_cameras_iterator)
                },
                self.rgb_compress_topic: {
                    'data_class': CompressedImage,
                    'callback': self._rgb_compress_callback,
                    'open': next(self.require_cameras_iterator)
                },
                self.depth_topic: {
                    'data_class': Image,
                    'callback': self._depth_callback,
                    'open': next(self.require_cameras_iterator)
                },
                self.depth_compress_topic: {
                    'data_class': CompressedImage,
                    'callback': self._depth_compress_callback,
                    'open': next(self.require_cameras_iterator)
                }
            }
        except StopIteration:
            raise ValueError(
                f"require_cameras value: {self.require_cameras} must equal to 4 which corresponding the:\
                \n\trgb_camera\n\trgb_compress_camera\n\tdepth_camera\n\tdepth_compress_camera"
            )

        for topic, config in self.cameras.items():
            if config['open']:
                rospy.Subscriber(
                    name=topic,
                    data_class=config['data_class'],
                    callback=config['callback']
                )
                rospy.loginfo(f'Waiting for {topic}')
                rospy.wait_for_message(topic, config['data_class'])
                rospy.loginfo(f'{topic}: OK')
                
    def __call__(self, image_id='rgb_compress'):
        return self.image_ids[image_id]

    def _rgb_callback(self, msg: Image):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_ids['rgb'] = self.rgb_image

    def _rgb_compress_callback(self, msg: CompressedImage):
        self.rgb_compress_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.image_ids['rgb_compress'] = self.rgb_compress_image

    def _depth_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        self.image_ids['depth'] = self.depth_image

    def _depth_compress_callback(self, msg: CompressedImage):
        self.depth_compress_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        self.image_ids['depth_compress'] = self.depth_compress_image
