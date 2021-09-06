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

from core.base_classes import Node
from core.Nodes.Visions import YOLODetection, vision_abstract

from home_robot_msgs.msg import VisionRequest
from home_robot_msgs.srv import VisionDeregister, VisionDeregisterRequest, VisionDeregisterResponse
from cv_bridge import CvBridge
import rospy

import threading


class VisionControllerNode(Node):
    def __init__(self, name: str = 'VC', anonymous: bool = False):
        # Model NodePrograms with id
        self.models = {
            'yolo': YOLODetection('yolo')
        }

        # Super the class
        super(VisionControllerNode, self).__init__(name, anonymous)

        # The goal subscriber
        self.requests_sub = rospy.Subscriber(
            f'{rospy.get_name()}/goal_requests',
            VisionRequest,
            self._request_cb,
            queue_size=10
        )

        # The deregister service
        self.deregister_srv = rospy.Service(
            f'{rospy.get_name()}/deregister',
            VisionDeregister,
            self._deregister_handler
        )

        # CvBridge initialize
        self.bridge = CvBridge()

        # All requests, operate with queue
        self.requests = []
        self._is_running = False

        # Client private topics with the publishers
        self.address_publishers = {}

        # Initialize the lock
        self.mutex = threading.RLock()

    def _deregister_handler(self, req: VisionDeregisterRequest):
        try:
            # Delete the private address
            del self.address_publishers[req.deregister_topic]
            success = True
        except KeyError:
            # The private address wasn't registered
            rospy.logerr(f"Can't deregister unregister private topic {req.deregister_topic}")
            success = False

        return VisionDeregisterResponse(ok=success)

    def _request_cb(self, client_request: VisionRequest):
        """
        This is the callback func for requests coming. All requests got here
        will be packed in a queue and executed with queue algorithm.
        Args:
            client_request: The request of the client, msg_type: VisionRequest
        Returns:

        """
        # Save client's request
        with self.mutex:
            self.requests.append(client_request)

        if not self._is_running:  # If the loop isn't running
            # Set it is running
            self._is_running = True
            # Pop out data from queue
            while len(self.requests) > 0:
                request: VisionRequest = self.requests.pop(0)

                # Get required data from request
                private_address = request.private_topic
                eval_model = request.eval_model
                input_image = request.input_image
                one_time = request.one_time

                # Get client name
                caller = request._connection_header['callerid']

                # Get model's NodeProgram from models, if None, log an error
                if eval_model not in self.models:
                    rospy.logerr(f"Model id not recognized, skipping client {caller}")
                    continue

                model: vision_abstract.VisionNodeProgram = self.models[eval_model]

                # If the private address is not recorded, then we'll record and create a publisher.
                # If it isn't provided, we'll skip this client
                if isinstance(private_address, str):
                    if private_address not in self.address_publishers:
                        self.address_publishers[private_address] = rospy.Publisher(
                            private_address,
                            model.output_msg,
                            queue_size=10
                        )
                else:
                    rospy.logerr(f"Client didn't provide it's address, skipping client {caller}")

                # Decode image format, if image is None, skip client
                input_image = self.bridge.compressed_imgmsg_to_cv2(input_image)
                # Get result from model and publish it to private address
                result = model.run(input_data=input_image, serialize=True)
                self.address_publishers[private_address].publish(result)

                # If client only wants to request one time, not live
                if one_time:
                    del self.address_publishers[private_address]

            self._is_running = False

    def reset(self):
        self.requests_sub.unregister()
        self.__init__()


if __name__ == '__main__':
    node = VisionControllerNode()
    node.spin()
