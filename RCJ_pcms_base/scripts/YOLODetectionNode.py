#!/usr/bin/env python3

import cv2 as cv
import rospy
import sensor_msgs.msg
from PIL import Image
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes
from home_robot_msgs.srv import ChangeImgSource, ChangeImgSourceResponse, ChangeImgSourceRequest
from rich.console import Console
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger, TriggerResponse

from core.Nodes import Node
from core.SensorFuncWrapper import YOLOV5

console = Console()
print = console.print


class YOLODetectionNode(Node):
    W = 640
    H = 480

    def __init__(self, node_name, anonymous=False):
        super(YOLODetectionNode, self).__init__(node_name, anonymous)

        with console.status("[magenta]Loading YOLO into program") as status:
            self.yolo = YOLOV5(rospy.get_param('~model_type'))

            image_source = rospy.get_param('~image_source')
            status.update(f'[bold yellow]Getting image from {rospy.get_param("~image_source")}', spinner='smiley')
            rospy.wait_for_message(image_source, CompressedImage)
            console.log(f"Camera topic ok")
        print("[bold cyan]YOLO loaded")

        self.boxes_pub = rospy.Publisher(
            '~boxes',
            ObjectBoxes,
            queue_size=1
        )

        self.drown_image_pub = rospy.Publisher(
            '~source_image/compressed',
            CompressedImage,
            queue_size=1
        )

        self.cam_sub = rospy.Subscriber(
            image_source,
            sensor_msgs.msg.CompressedImage,
            self.image_callback,
            queue_size=1
        )

        self.change_request = rospy.Service(
            '~change_camera',
            ChangeImgSource,
            self.change_camera
        )
        rospy.Service(
            '~lock',
            Trigger,
            self.lock_callback
        )
        rospy.Service(
            '~kill',
            Trigger,
            self.kill_callback
        )
        rospy.Service(
            '~publish_drown_image',
            Trigger,
            self.publish_drown_image_callback
        )

        self._isLock = False
        self._isKill = False
        self._isPublishDrownImage = False

        self.bridge = CvBridge()

        self._srcframe = None

    def image_callback(self, image: sensor_msgs.msg.CompressedImage):
        input_image = self.bridge.compressed_imgmsg_to_cv2(image)
        if rospy.get_param('~reverse'):
            input_image = cv.flip(input_image, 0)
        H, W, _ = input_image.shape

        YOLODetectionNode.H = H
        YOLODetectionNode.W = W

        self._srcframe = input_image

    def change_camera(self, new_cam_topic: ChangeImgSourceRequest):
        self.cam_sub.unregister()
        new_topic = new_cam_topic.new_topic

        try:
            _ = rospy.wait_for_message(new_topic, Image, timeout=5)
            self.cam_sub = rospy.Subscriber(
                new_topic,
                Image,
                self.image_callback,
                queue_size=1
            )
            rospy.loginfo(f'Changed new image source to {new_topic}')
            ok = True
        except rospy.exceptions.ROSException:
            ok = False
            self._srcframe = None

        return ChangeImgSourceResponse(ok=ok)

    def lock_callback(self, data):
        self._isLock = not self._isLock
        return TriggerResponse(success=True, message="YOLO successfully locked")

    def kill_callback(self, data):
        self._isKill = not self._isKill
        return TriggerResponse(success=True, message="YOLO successfully killed")

    def publish_drown_image_callback(self, data):
        self._isPublishDrownImage = not self._isPublishDrownImage
        will_or_not = 'will' if self._isPublishDrownImage else 'will not'
        return TriggerResponse(success=True, message=f"YOLO {will_or_not} publish drown image starting from now on")

    def main(self):
        while not rospy.is_shutdown() and not self._isKill:
            if not self._isLock:
                if self._srcframe is None:
                    continue

                drown_image = self._srcframe.copy()
                frame = drown_image.copy()

                results = self.yolo.detect(drown_image)
                results.render()
                boxes = self.yolo.serialize(frame, results)

                self.boxes_pub.publish(boxes)
                if self._isPublishDrownImage:
                    self.drown_image_pub.publish(drown_image)

                cv.imshow('YD', drown_image)
                key = cv.waitKey(1) & 0xFF
                if key in [27, ord('q')]:
                    break

            self.rate.sleep()
        cv.destroyAllWindows()

    def reset(self) -> TriggerResponse:
        return super(YOLODetectionNode, self).reset()


if __name__ == "__main__":
    node = YOLODetectionNode('YD')
    node.main()
