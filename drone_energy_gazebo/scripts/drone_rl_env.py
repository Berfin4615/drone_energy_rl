# #!/usr/bin/env python3

# import rospy
# import numpy as np
# from gazebo_msgs.msg import LinkStates
# from geometry_msgs.msg import Twist
# from hector_uav_msgs.msg import Supply
# from std_srvs.srv import Empty

# """
# Drone Start: (-62.94, 62.93, 1.0)
#        ▲
#        │
#        │ 125m horizontal distance
#        │
#        ▼
# Goal Box: (62.51, -62.23, 0.55)
#      +-------------------+
#      |                   |
#      |    Landing Zone   |
#      |                   |
#      +-------------------+
# """

# class DroneRLEnv:
#     def __init__(self):
#         self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
#         rospy.Subscriber('/supply', Supply, self._battery_cb)
#         rospy.Subscriber('/gazebo/link_states', LinkStates, self._gazebo_cb)

#         # Environment state
#         self.battery = 100.0
#         self.altitude = 0.0

#         self.initial_position = np.array([-62.944, 62.9287, 1.0])


#         #TODO!: Is velocity necessary?
#         self.velocity = np.zeros(2)
#         self.current_position = self.initial_position.copy()
#         self.goal_position = np.array([62.5097, -62.2278, 0.55])
#         self.energy_consumed = 0.0
#         self.base_energy_rate = 0.1

#         rospy.sleep(1.0)  # Allow time for topics to initialize

#     def _battery_cb(self, msg):
#         self.battery = msg.percentage

#     def _gazebo_cb(self, msg):
#         try:
#             idx = msg.name.index('hector_quadrotor::base_link')
#             pos = msg.pose[idx].position
#             self.altitude = pos.z
#             self.current_position = np.array([pos.x, pos.y, pos.z])
#             self.velocity = np.array([
#                 msg.twist[idx].linear.x,
#                 msg.twist[idx].linear.y
#             ])
#         except ValueError:
#             pass  # Handle if model not yet spawned

#     def _get_state(self):
#         horizontal_distance = np.linalg.norm(self.goal_position[:2] - self.current_position[:2])
#         vertical_distance = abs(self.goal_position[2] - self.current_position[2])
#         distance_to_goal = np.linalg.norm(self.goal_position[:2] - self.current_position[:2])
#         return np.array([
#             self.battery,
#             self.current_position[2],  # Current altitude
#             horizontal_distance,
#             vertical_distance,  # Added vertical distance
#             self.velocity[0],
#             self.velocity[1]
#         ], dtype=np.float32)
# #TODO: Are horizontal and vertical distance important or just the total distance? 
# #TODO: Voltage is 14.0 when full 10.0 when empty, should we use this instead of battery percentage?
#     def _calculate_reward(self, horizontal_dist, vertical_dist, energy_step):
#         landed = (horizontal_dist < 2.0) and (vertical_dist < 0.3)  # Box is 4x4m
#         goal_reward = 100.0 if landed else -0.1 * (horizontal_dist + vertical_dist)
#         energy_penalty = -0.5 * energy_step
#         return goal_reward + energy_penalty


#     def step(self, action):
#         # 1. Publish command
#         cmd = Twist()
#         cmd.linear.x, cmd.linear.y, cmd.linear.z = action[:3]
#         cmd.angular.z = action[3]
#         self.cmd_vel_pub.publish(cmd)

#         rospy.sleep(0.1)  # Let simulation update

#         # 2. Get updated state
#         state = self._get_state()
#         self.altitude = state[1]
#         velocity_magnitude = np.linalg.norm(action[:3])

#         # 3. Compute energy consumption
#         energy_step = self.base_energy_rate * (1 + velocity_magnitude) * (1 + abs(self.altitude))
#         self.energy_consumed += energy_step

#         self.battery -= energy_step
#         self.battery = max(self.battery, 0.0)

#         # 4. Reward and done condition
#         distance = state[2]
#         reward = self._calculate_reward(distance, energy_step)
#         horizontal_dist = state[2]
#         vertical_dist = state[3]
#         done = (self.battery <= 0.0) or (horizontal_dist < 2.0 and vertical_dist < 0.3)

#         # 5. Info for logging
#         info = {
#             "battery": self.battery,
#             "altitude": self.altitude,
#             "velocity": velocity_magnitude,
#             "distance_to_goal": distance,
#             "energy_step": energy_step,
#             "total_energy": self.energy_consumed,
#             "reward": reward
#         }
#         truncated = False
#         return state, reward, done, truncated, info

#     def reset(self):
#         rospy.wait_for_service('/gazebo/reset_world')
#         try:
#             reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
#             reset_world()
#             self.current_position = self.initial_position.copy()
#             self.altitude = self.initial_position[2]
#         except rospy.ServiceException as e:
#             rospy.logerr("Failed to reset simulation: %s", e)

#         self.energy_consumed = 0.0
#         self.battery = 100.0
#         rospy.sleep(1.0)  # Wait for reset to take effect
#         return self._get_state()

#     def close(self):
#         rospy.loginfo("Shutting down DroneRLEnv.")

#!/usr/bin/env python3

import rospy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Twist
from hector_uav_msgs.msg import Supply
from std_srvs.srv import Empty

class DroneRLEnv(gym.Env):
    def __init__(self):
        super(DroneRLEnv, self).__init__()
        
        # Initialize ROS components
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/supply', Supply, self._supply_cb)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self._gazebo_cb)
        
        # Battery voltage parameters
        self.full_voltage = 14.8
        self.empty_voltage = 10.0
        
        # Define landscape boundaries (based on your 120m landscape)
        self.x_min = -60.0  # -60m
        self.x_max = 60.0   # +60m
        self.y_min = -60.0  # -60m
        self.y_max = 60.0   # +60m
        self.z_min = 0.1    # Minimum altitude (above ground)
        self.z_max = 50.0   # Maximum altitude
        
        # Define discrete action space for DQN
        self.action_space = spaces.Discrete(9)
        
        # Action mapping: [linear.x, linear.y, linear.z, angular.z]
        self.action_mapping = {
            0: [0.0, 0.0, 0.0, 0.0],    # Hover
            1: [1.0, 0.0, 0.0, 0.0],    # Forward
            2: [-1.0, 0.0, 0.0, 0.0],   # Backward
            3: [0.0, 1.0, 0.0, 0.0],    # Right
            4: [0.0, -1.0, 0.0, 0.0],   # Left
            5: [0.0, 0.0, 1.0, 0.0],    # Up
            6: [0.0, 0.0, -1.0, 0.0],   # Down
            7: [0.0, 0.0, 0.0, 1.0],    # Rotate right
            8: [0.0, 0.0, 0.0, -1.0]    # Rotate left
        }

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -10, -10]),
            high=np.array([100, 100, 200, 100, 10, 10]),
            dtype=np.float32
        )

        # Environment state
        self.battery_voltage = self.full_voltage
        self.battery_percentage = 100.0
        self.altitude = 0.0
        self.velocity = np.zeros(2)
        self.current_position = np.array([-62.944, 62.9287, 1.0])
        self.goal_position = np.array([62.5097, -62.2278, 0.55])
        self.energy_consumed = 0.0
        self.base_energy_rate = 0.1

        rospy.sleep(1.0)

    def _supply_cb(self, msg):
        # Handle voltage list
        if len(msg.voltage) > 0:
            self.battery_voltage = msg.voltage[0]
            # Convert voltage to percentage
            voltage_range = self.full_voltage - self.empty_voltage
            self.battery_percentage = max(0.0, min(100.0, 
                (self.battery_voltage - self.empty_voltage) / voltage_range * 100.0
            ))

    def _gazebo_cb(self, msg):
        try:
            idx = msg.name.index('hector_quadrotor::base_link')
            pos = msg.pose[idx].position
            self.altitude = pos.z
            self.current_position = np.array([pos.x, pos.y, pos.z])
            self.velocity = np.array([
                msg.twist[idx].linear.x,
                msg.twist[idx].linear.y
            ])
        except ValueError:
            pass

    def _get_state(self):
        horizontal_distance = np.linalg.norm(self.goal_position[:2] - self.current_position[:2])
        vertical_distance = abs(self.goal_position[2] - self.current_position[2])
        return np.array([
            self.battery_percentage,
            self.current_position[2],
            horizontal_distance,
            vertical_distance,
            self.velocity[0],
            self.velocity[1]
        ], dtype=np.float32)

    def _calculate_reward(self, horizontal_dist, vertical_dist, energy_step):
        landed = (horizontal_dist < 2.0) and (vertical_dist < 0.3)
        goal_reward = 100.0 if landed else -0.1 * (horizontal_dist + vertical_dist)
        energy_penalty = -0.5 * energy_step
        return goal_reward + energy_penalty

    def _is_out_of_bounds(self):
        """Check if drone is outside landscape boundaries"""
        x, y, z = self.current_position
        return (x < self.x_min or x > self.x_max or 
                y < self.y_min or y > self.y_max or 
                z < self.z_min or z > self.z_max)

    def step(self, action_idx):
        # Map discrete action to continuous command
        action = self.action_mapping[action_idx]
        
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z, cmd.angular.z = action
        self.cmd_vel_pub.publish(cmd)

        rospy.sleep(0.1)

        state = self._get_state()
        horizontal_dist = state[2]
        vertical_dist = state[3]
        velocity_magnitude = np.linalg.norm(action[:3])

        energy_step = self.base_energy_rate * (1 + velocity_magnitude) * (1 + abs(self.altitude))
        self.energy_consumed += energy_step
        self.battery_percentage = max(self.battery_percentage - energy_step, 0.0)

        # Check if drone is out of bounds
        out_of_bounds = self._is_out_of_bounds()
        
        if out_of_bounds:
            reward = -100  # Large penalty for leaving the environment
            terminated = True
            truncated = True
        else:
            reward = self._calculate_reward(horizontal_dist, vertical_dist, energy_step)
            terminated = (self.battery_percentage <= 0.0) or (horizontal_dist < 2.0 and vertical_dist < 0.3)
            truncated = False
        
        info = {
            "battery": self.battery_percentage,
            "altitude": self.altitude,
            "velocity": velocity_magnitude,
            "horizontal_distance": horizontal_dist,
            "vertical_distance": vertical_dist,
            "energy_step": energy_step,
            "total_energy": self.energy_consumed,
            "out_of_bounds": out_of_bounds
        }
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_world()
        except rospy.ServiceException as e:
            rospy.logerr("Reset failed: %s", e)

        self.energy_consumed = 0.0
        self.battery_voltage = self.full_voltage
        self.battery_percentage = 100.0
        
        # Reset to initial position
        self.current_position = np.array([-62.944, 62.9287, 1.0])
        
        rospy.sleep(1.0)
        return self._get_state(), {}

    def close(self):
        rospy.loginfo("Shutting down DroneRLEnv.")