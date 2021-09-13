from abc import ABC

import rospy


class Tools(ABC):
    @classmethod
    def _check_status(cls):
        if not rospy.core.is_initialized():
            raise rospy.ROSException('Please initialize first')
