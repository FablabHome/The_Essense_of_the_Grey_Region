import cv2 as cv
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


class TrueCoordinatesTest:
    # Camera uplifted angle
    CAMERA_ANGLE = 17

    # Field of view of astra camera
    FOV_H = 60
    FOV_V = 49.5

    # Camera W and H
    W = 640
    H = 480

    def __init__(self):
        self.rgb_image = self.depth_image = None
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)

        rospy.Subscriber(
            '/camera/rgb/image_raw/compressed',
            CompressedImage,
            self.rgb_callback,
            queue_size=1
        )
        rospy.Subscriber(
            '/camera/depth/image_raw',
            Image,
            self.depth_callback,
            queue_size=1
        )

    def rgb_callback(self, image: CompressedImage):
        self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(image)

    def depth_callback(self, depth: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(depth)

    @staticmethod
    def angle_2_radian(angle):
        return (np.pi * angle) / 180

    @classmethod
    def reverse_l2_cross(cls, ox, oy):
        hw = cls.W / 2
        hh = cls.H / 2
        cross_x = ox - hw
        cross_y = hh - oy
        return cross_x, cross_y

    def main(self):
        init_box = None
        while not rospy.is_shutdown():
            if self.rgb_image is None or self.depth_image is None:
                continue

            rgb_image = self.rgb_image.copy()
            depth_image = self.depth_image.copy()

            if init_box is not None:
                x, real_y, w, h = init_box
                cv.rectangle(rgb_image, (x, real_y), (x + w, real_y + h), (32, 0, 255), 6)

                cx = min(640, (w // 2) + x)
                cy = min(480, (h // 2) + real_y)
                cz = depth_image[cy, cx]
                cv.circle(rgb_image, (cx, cy), 5, (32, 255, 0), -1)

                # Reverse l 2z cross
                # cx, cy = self.reverse_l2_cross(cx, cy)

                # real_y = cz * sin(self.angle_2_radian(TrueCoordinatesTest.A_THETA)) + (cy / 480 - 1) * cz * tan(self.angle_2_radian(49.5 / 2)) * cos(self.angle_2_radian(TrueCoordinatesTest.A_THETA))
                rad_h = self.angle_2_radian(TrueCoordinatesTest.FOV_H / 2)
                rad_v = self.angle_2_radian(TrueCoordinatesTest.FOV_V / 2)
                rad_cam_angle = self.angle_2_radian(TrueCoordinatesTest.CAMERA_ANGLE)
                real_w = 2 * cz * np.tan(rad_h)
                real_h = 2 * cz * np.tan(rad_v)

                # Real x
                real_x = real_w * cx / TrueCoordinatesTest.W
                real_x -= (real_w / 2)

                # Real y
                real_y = real_h * cy / TrueCoordinatesTest.H
                FE = cz * np.sin(rad_cam_angle)
                GF = (0.5 * real_h - real_y) * np.cos(rad_cam_angle)
                final_y = FE + GF

                # Real z
                OD = cz * np.cos(rad_cam_angle)
                ED = GF * np.tan(rad_cam_angle)
                real_z = OD - ED

                info_text = f'original: {(cx, cy, cz)}, reality: {real_x:.2f}, {final_y:.2f}, {real_z:.2f}'
                cv.putText(rgb_image, info_text, (3, TrueCoordinatesTest.H - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (32, 255, 0), 2)

            cv.imshow('frame', rgb_image)
            key = cv.waitKey(1) & 0xFF

            if key in [ord('q'), 27]:
                break
            elif key == ord('c'):
                init_box = cv.selectROI('frame', rgb_image)

            self.rate.sleep()

        cv.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('true_distance')
    node = TrueCoordinatesTest()
    node.main()
