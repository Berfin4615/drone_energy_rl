#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import BatteryState
import csv

class EnergyLogger:
    def __init__(self):
        rospy.Subscriber('/battery', BatteryState, self.log_data)
        self.csv_file = open('energy_log.csv', 'w')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(['Time', 'Battery%', 'Voltage', 'Current'])

    def log_data(self, msg):
        self.writer.writerow([
            rospy.get_time(),
            msg.percentage,
            msg.voltage,
            msg.current
        ])

if __name__ == '__main__':
    rospy.init_node('energy_logger')
    logger = EnergyLogger()
    rospy.spin()