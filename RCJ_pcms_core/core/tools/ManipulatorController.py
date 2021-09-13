"""
MIT License

Copyright (c) 2020 rootadminWalker

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import rospy
from open_manipulator_msgs.srv import SetKinematicsPose, SetJointPosition, SetKinematicsPoseRequest, \
    SetJointPositionRequest

from .Abstract import Tools


class ManipulatorController(Tools):
    MANI_SRV_NAME = '/goal_task_space_path_position_only'
    MANI_GRIPPER_SRV_NAME = '/goal_tool_control'
    MANI_JOINT_SRV_NAME = '/goal_joint_space_path'

    INIT_POSE = (0.290, 0.000, 0.193)
    HOME_POSE = (0.134, 0.000, 0.240)

    INIT_JOINT = (0.0, 0.021, 0.061, 0.018)
    HOME_JOINT = (0.0, -1.040, 0.370, 0.706)

    def __init__(self):
        super()._check_status()

        rospy.wait_for_service(ManipulatorController.MANI_SRV_NAME)
        self.task_space_control = rospy.ServiceProxy(ManipulatorController.MANI_SRV_NAME, SetKinematicsPose)

        rospy.wait_for_service(ManipulatorController.MANI_GRIPPER_SRV_NAME)
        self.gripper_control = rospy.ServiceProxy(ManipulatorController.MANI_GRIPPER_SRV_NAME, SetJointPosition)

        rospy.wait_for_service(ManipulatorController.MANI_JOINT_SRV_NAME)
        self.joint_control = rospy.ServiceProxy(ManipulatorController.MANI_JOINT_SRV_NAME, SetJointPosition)

    def move_to(self, x, y, z, t):
        try:
            req = SetKinematicsPoseRequest()
            req.end_effector_name = 'gripper'
            req.kinematics_pose.pose.position.x = x
            req.kinematics_pose.pose.position.y = y
            req.kinematics_pose.pose.position.z = z
            req.path_time = t

            resp = self.task_space_control(req)
            return resp
        except Exception as e:
            rospy.loginfo(e)

    def set_gripper(self, position, t):
        try:
            req = SetJointPositionRequest()
            req.joint_position.joint_name = ["gripper"]
            req.joint_position.position = [position]
            req.path_time = t
            resp = self.gripper_control(req)
            return resp
        except Exception as e:
            rospy.loginfo(e)

    def set_joints(self, joint1, joint2, joint3, joint4, t):
        try:
            req = SetJointPositionRequest()
            req.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
            req.joint_position.position = [joint1, joint2, joint3, joint4]
            req.path_time = t
            resp = self.joint_control(req)
            return resp
        except Exception as e:
            rospy.loginfo(e)
