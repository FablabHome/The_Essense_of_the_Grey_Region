import rospy
from core.tools import Chassis

rospy.init_node('test_chassis')
chassis = Chassis()
rospy.sleep(1)
# chassis.turn(3.14)
chassis.check_yaw()
