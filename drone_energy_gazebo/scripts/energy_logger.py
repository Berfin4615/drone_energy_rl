#!/usr/bin/env python3
import rospy
import csv
import time
from sensor_msgs.msg import BatteryState
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Vector3Stamped

class EnergyLogger:
    def __init__(self):
        rospy.init_node('energy_logger')
        
        # Data storage
        self.data = []
        
        # Subscribers
        rospy.Subscriber('/battery', BatteryState, self.battery_cb)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.pose_cb)
        rospy.Subscriber('/wind', Vector3Stamped, self.wind_cb)
        
        # Create CSV file
        self.filename = f"energy_log_{int(time.time())}.csv"
        with open(self.filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'battery%', 'voltage', 'current', 
                'position_x', 'position_y', 'position_z',
                'velocity_x', 'velocity_y', 'velocity_z',
                'wind_x', 'wind_y'
            ])
    
    def battery_cb(self, msg):
        self.battery = msg.percentage
        self.voltage = msg.voltage
        self.current = msg.current
    
    def pose_cb(self, msg):
        try:
            idx = msg.name.index('quadrotor::base_link')
            self.position = msg.pose[idx].position
            self.velocity = msg.twist[idx].linear
        except ValueError:
            pass
    
    def wind_cb(self, msg):
        self.wind = msg.vector
        
        # Log all available data
        with open(self.filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                rospy.get_time(),
                getattr(self, 'battery', 0),
                getattr(self, 'voltage', 0),
                getattr(self, 'current', 0),
                getattr(self, 'position', type('',(object,),{'x':0,'y':0,'z':0})).x,
                getattr(self, 'position', type('',(object,),{'x':0,'y':0,'z':0})).y,
                getattr(self, 'position', type('',(object,),{'x':0,'y':0,'z':0})).z,
                getattr(self, 'velocity', type('',(object,),{'x':0,'y':0,'z':0})).x,
                getattr(self, 'velocity', type('',(object,),{'x':0,'y':0,'z':0})).y,
                getattr(self, 'velocity', type('',(object,),{'x':0,'y':0,'z':0})).z,
                self.wind.x,
                self.wind.y
            ])

if __name__ == '__main__':
    logger = EnergyLogger()
    rospy.spin()