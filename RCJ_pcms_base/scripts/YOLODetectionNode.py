#!/usr/bin/env python3

import os

import cv2 as cv
import numpy as np
import rospy
import sensor_msgs.msg
from PIL import Image
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBox, ObjectBoxes
from home_robot_msgs.srv import ChangeImgSource, ChangeImgSourceResponse, ChangeImgSourceRequest
from keras_yolo3.yolo import YOLO_np
from rospkg import RosPack
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool, SetBoolResponse


class YOLODetectionNode:
    W = 640
    H = 480

    def __init__(self):
        rospy.loginfo(f'Getting image from {rospy.get_param("~image_source")}')
        self.source_image_pub = rospy.Publisher(
            '~source_image/compressed',
            CompressedImage,
            queue_size=1
        )

        self.cam_sub = rospy.Subscriber(
            rospy.get_param('~image_source', '/camera/rgb/image_raw'),
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
            SetBool,
            self.lock_callback
        )
        rospy.Service(
            '~kill',
            SetBool,
            self.kill_callback
        )

        self.lock = False
        self.kill = False

        self.bridge = CvBridge()

        self.source_image = self.blob = None

    def image_callback(self, image: sensor_msgs.msg.CompressedImage):
        input_image = self.bridge.compressed_imgmsg_to_cv2(image)
        if rospy.get_param('~reverse'):
            input_image = cv.flip(input_image, 0)
        H, W, _ = input_image.shape

        YOLODetectionNode.H = H
        YOLODetectionNode.W = W

        self.source_image = input_image
        self.blob = Image.fromarray(input_image)

        # self.net_yolo.setInput(blob)
        # outputs = self.net_yolo.forward()

        # self.objects = DetectBox.parse_output(outputs, W, H)

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
            self.objects = []
            self.source_image = None

        return ChangeImgSourceResponse(ok=ok)

    def lock_callback(self, data):
        self.lock = data.data
        return SetBoolResponse()

    def kill_callback(self, data):
        self.kill = data.data
        return SetBoolResponse()


if __name__ == "__main__":
    rospy.init_node('YD')
    base = RosPack().get_path('rcj_pcms_base') + '/..'
    _model_h5 = os.path.join(base, 'models/YOLO/yolov3.h5')
    _coco_classes = os.path.join(base, 'models/YOLO/coco_classes.txt')
    _anchors = os.path.join(base, 'models/YOLO/yolo3_anchors.txt')
    _model = YOLO_np(
        model_type='yolo3_darknet',
        yolo_weights_path='',
        weights_path=_model_h5,
        anchors_path=_anchors,
        classes_path=_coco_classes
    )

    classnames = open(_coco_classes).readlines()

    box = ObjectBox()

    boxes = ObjectBoxes()

    node = YOLODetectionNode()
    rate = rospy.Rate(35)

    pub = rospy.Publisher(
        '~boxes',
        ObjectBoxes,
        queue_size=1
    )

    while not rospy.is_shutdown() and not node.kill:
        if not node.lock:
            box_items = []
            boxes = ObjectBoxes()
            # Get objects and source images
            # objects = node.objects
            blob = node.blob
            source_image = node.source_image

            if source_image is None or blob is None:
                continue

            drown_image, out_boxes, out_classes, out_scores = _model.detect_image(blob)
            drown_image = np.array(drown_image)

            for obj, label, score in zip(out_boxes, out_classes, out_scores):
                x1, y1, x2, y2 = obj
                if score < 0.45:
                    continue

                box = ObjectBox()
                # Input box data
                box.model = 'yolo'
                box.x1 = x1
                box.y1 = y1
                box.x2 = x2
                box.y2 = y2
                box.score = score
                box.label = classnames[label]

                box_image = source_image[box.y1:box.y2, box.x1:box.x2].copy()

                # Serialize the image
                serialized_image = node.bridge.cv2_to_compressed_imgmsg(box_image)
                box.source_img = serialized_image

                box_items.append(box)

            boxes.boxes = box_items
            boxes.source_img = node.bridge.cv2_to_compressed_imgmsg(source_image)

            # serialized_drown_img = node.bridge.cv2_to_compressed_imgmsg(np.array(drown_image))
            # node.source_image_pub.publish(serialized_drown_img)
            pub.publish(boxes)

            try:
                cv.imshow('YD', drown_image)
                key = cv.waitKey(1) & 0xFF
                if key in [27, ord('q')]:
                    break
            except Exception:
                rospy.loginfo(drown_image.shape)
            rate.sleep()

    cv.destroyAllWindows()
