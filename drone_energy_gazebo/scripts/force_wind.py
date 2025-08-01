#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench
import time
from geometry_msgs.msg import Vector3

def set_wind():
    rospy.init_node('force_wind')
    rospy.wait_for_service('/gazebo/apply_body_wrench')
    
    try:
        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        response = apply_wrench(
            body_name="quadrotor::base_link",
            wrench=Wrench(
                force=Vector3(2.0, 1.0, 0.0),
                torque=Vector3(0,0,0)
            ),
            duration=rospy.Duration(-1)  # Continuous force
        )
        rospy.loginfo("Wind applied successfully!")
    except rospy.ServiceException:
        rospy.logerr("Failed to apply wind")

if __name__ == '__main__':
    set_wind()