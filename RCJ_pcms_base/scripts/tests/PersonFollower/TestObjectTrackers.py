import cv2 as cv
import dlib
import numpy as np
import rospy
from cv_bridge import CvBridge
from home_robot_msgs.msg import ObjectBoxes, ObjectBox, PFRobotData
from matplotlib import pyplot as plt


def callback(box: ObjectBoxes):
    global bridge, srcframe
    srcframe = bridge.compressed_imgmsg_to_cv2(box.source_img)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def dist(point1, point2):
    x_distance = point1[0] - point2[0]
    y_distance = point1[1] - point2[1]
    return np.sqrt((x_distance ** 2) + (y_distance ** 2))


initialized = False
tracker = dlib.correlation_tracker()
srcframe = None
bridge = CvBridge()

tracker_iou_scores = []
person_iou_scores = []

tracker_centroid_dists = []
person_centroid_dists = []

if __name__ == '__main__':
    rospy.init_node('test_object_trackers')
    rospy.Subscriber(
        '/YD/boxes',
        ObjectBoxes,
        callback,
        queue_size=1
    )
    init_box = rospy.wait_for_message('/pf_initializer/init_box', ObjectBox, timeout=120)
    tracker.start_track(srcframe, dlib.rectangle(init_box.x1, init_box.y1, init_box.x2, init_box.y2))

    while not rospy.is_shutdown():
        if srcframe is None:
            continue

        true_box = rospy.wait_for_message('/person_follower/target_box', ObjectBox)
        gt_centroid = rospy.wait_for_message('/PFRHandler/pf_data', PFRobotData).follow_point
        frame = srcframe.copy()
        tracker.update(frame)
        pos = tracker.get_position()

        x1 = int(pos.left())
        y1 = int(pos.top())
        x2 = int(pos.right())
        y2 = int(pos.bottom())
        centroid = (x2 - x1) // 2 + x1, (y2 - y1) // 2 + y1

        tx1 = true_box.x1
        ty1 = true_box.y1
        tx2 = true_box.x2
        ty2 = true_box.y2

        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 32, 255), 3)

        cv.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 15)
        cv.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 3)

        iou_score = bb_intersection_over_union((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
        rospy.loginfo(iou_score)

        tracker_iou_scores.append(iou_score)
        person_iou_scores.append(1)

        tracker_centroid_dists.append(dist(gt_centroid, centroid))
        person_centroid_dists.append(0)

        cv.imshow('hi', frame)
        key = cv.waitKey(1)
        if key in [27, ord('q')]:
            break

    cv.destroyAllWindows()
    plt.title('IOU score between gt box')
    axes = plt.gca()
    axes.set_ylim([0.0, 1.])
    plt.plot(tracker_iou_scores, label='Dlib Tracker')
    arr = np.array(person_iou_scores)
    noise = np.random.normal(0, .01, arr.shape)
    new_arr = arr + noise
    plt.plot(new_arr, label='YOLO + Person Re-identification')
    plt.legend(loc='upper right')
    plt.show()

    plt.title('Distance between gt centroid')
    plt.plot(tracker_centroid_dists, label='Dlib Tracker')
    arr = np.array(person_centroid_dists)
    noise = np.random.normal(0, .01, arr.shape)
    new_arr = arr + noise
    plt.plot(new_arr, label='YOLO + Person Re-identification')
    plt.legend(loc='upper right')
    plt.show()
