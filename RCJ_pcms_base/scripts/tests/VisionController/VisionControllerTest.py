from core.Dtypes.boxProcess import deserialize_ros_to_bbox

from home_robot_msgs.msg import VisionRequest, ObjectBoxes
from core.tools import Camera
from cv_bridge import CvBridge
import rospy

import cv2 as cv


def callback(msg: ObjectBoxes):
    global boxes, _get_boxes
    boxes = deserialize_ros_to_bbox(msg)
    _get_boxes = True


rospy.init_node('test_VC')
pub = rospy.Publisher('/VC/goal_requests', VisionRequest, queue_size=10)
rospy.Subscriber('/here/my_family', ObjectBoxes, callback, queue_size=10)

cap = Camera(require_cameras=(0, 1, 0, 0), rgb_topic='/image_raw')
boxes = []

bridge = CvBridge()

frame = None

_get_image = True
_get_boxes = True

while not rospy.is_shutdown():
    if _get_image:
        frame = cap('rgb_compress')
        _get_image = False

    if _get_boxes:
        if frame is not None:
            compressed_frame = bridge.cv2_to_compressed_imgmsg(frame)
            pub.publish(VisionRequest(input_image=compressed_frame, private_topic='/here/my_family', eval_model='yolo'))
            for box in boxes:
                box.draw(frame)

            cv.imshow('frame', frame)
            cv.waitKey(16)
        _get_boxes = False
        _get_image = True
