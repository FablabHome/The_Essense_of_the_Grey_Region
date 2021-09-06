import cv2 as cv
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes
import matplotlib


def callback(boxes: ObjectBoxes):
    global persons
    persons = list(filter(lambda x: x.label.strip() == 'person', boxes.boxes))


plt = matplotlib.pyplot
matplotlib.use('TkAgg')
persons = []
bridge = CvBridge()
if __name__ == '__main__':
    rospy.init_node('test_backlight')
    rospy.Subscriber(
        '/YD/boxes',
        ObjectBoxes,
        callback,
        queue_size=1
    )
    while not rospy.is_shutdown():
        for person in persons:
            person_img = bridge.compressed_imgmsg_to_cv2(person.source_img)
            gray = cv.cvtColor(person_img, cv.COLOR_BGR2GRAY)
            plt.hist(gray.ravel(), 256, [0, 256])
            plt.show()
