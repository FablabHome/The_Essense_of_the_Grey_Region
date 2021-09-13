#!/usr/bin/env python3

import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from pyzbar import pyzbar
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

rospy.init_node('qr_code')


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def callback(img: CompressedImage):
    global bridge, srcframe
    srcframe = bridge.compressed_imgmsg_to_cv2(img)


rospy.set_param('~lock', True)
pub = rospy.Publisher(
    '~status',
    String,
    queue_size=1
)

speaker_pub = rospy.Publisher(
    '/speaker/say',
    String,
    queue_size=1
)


rospy.Subscriber(
    '/bottom_camera_rotate/rgb/image_raw/compressed',
    CompressedImage,
    callback,
    queue_size=1
)


def read_barcodes(frame):
    barcodes = pyzbar.decode(frame)
    crop_img2 = None
    for barcode in barcodes:
        x, y, w, h = barcode.rect

        barcode_info = barcode.data.decode('utf-8')
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img2 = frame[y:y + h, x:x + w]
        font = cv2.FONT_HERSHEY_DUPLEX

    return frame, crop_img2


def detect_color(img):
    try:
        valid = True
        height, width, _ = img.shape
        avg_color_per_row = np.average(img, axis=0)

        avg_colors = np.average(avg_color_per_row, axis=0)

        int_averages = np.array(avg_colors, dtype=np.uint8)
        print(f'int_averages: {int_averages}')
        if int_averages[-1] >= int_averages[1]:
            valid = False
        return valid
    except ZeroDivisionError:
        return None


srcframe = None
bridge = CvBridge()


def main():
    while not rospy.is_shutdown():
        if srcframe is None:
            continue

        frame = srcframe.copy()
        img_y, img_x, c = frame.shape
        x = int(img_x / 2 - 160)
        y = int(img_y / 2 - 160)
        w = 320
        h = 320
        crop_img = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        crop_img, crop_img2 = read_barcodes(crop_img)
        frame[y:y + h, x:x + w] = crop_img
        cv2.imshow('Barcode/QR code', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if crop_img2 is None:
            continue
        valid = detect_color(crop_img2)
        rospy.loginfo(unique_count_app(crop_img2))

        if valid:
            cv2.putText(frame, "Valid", (int(img_y / 2 - 60), int(img_y / 2 + 200)), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 255, 0), 5, cv2.LINE_AA, False)
            cv2.imshow('Barcode/QR code', frame)
            cv2.waitKey(3)
            time.sleep(3)
            pub.publish("Valid")
        else:
            cv2.putText(frame, "Invalid", (int(img_y / 2 - 60), int(img_y / 2 + 200)), cv2.FONT_HERSHEY_SIMPLEX,
                        3,
                        (0, 0, 255), 5, cv2.LINE_AA, False)
            pub.publish('Invalid')
            cv2.imshow('Barcode/QR code', frame)
            cv2.waitKey(3)
            time.sleep(3)

        if not rospy.get_param('~lock'):
            if not valid:
                speaker_pub.publish('Mister, Your health code is invalid')
            else:
                speaker_pub.publish('Your health code is valid')

#
if __name__ == '__main__':
    main()
