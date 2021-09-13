import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

rospy.init_node('test')
bridge = CvBridge()

while not rospy.is_shutdown():
      srcframe = rospy.wait_for_message('/bottom_camera/rgb/image_raw/compressed', CompressedImage)
      frame = bridge.compressed_imgmsg_to_cv2(srcframe)
      frame = cv2.flip(frame, 0)

      original = frame.copy()
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      lower = np.array([22, 93, 0], dtype="uint8")
      upper = np.array([45, 255, 255], dtype="uint8")
      mask = cv2.inRange(image, lower, upper)
      cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      cnts = cnts[0] if len(cnts) == 2 else cnts[1]

      for c in cnts:
          x, y, w, h = cv2.boundingRect(c)
          if w*h > 1000:
              print(w * h)
              cv2.rectangle(original, (x, y), (x + w, y + h), (36, 255, 12), 2)

      cv2.imshow('mask', mask)
      cv2.imshow('original', original)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

# cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
