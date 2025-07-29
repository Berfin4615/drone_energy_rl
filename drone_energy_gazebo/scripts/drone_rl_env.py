#!/usr/bin/env python3
import rospy
import numpy as np
import gym
from gym import spaces
from gazebo_msgs.msg import ModelState, LinkStates
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Float32
from sensor_msgs.msg import BatteryState

class DroneRLEnv(gym.Env):
    def __init__(self):
        super(DroneRLEnv, self).__init__()
        
        # Action space: [linear_x, linear_y, linear_z, angular_z] (normalized [-1, 1])
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        # State space: [battery_remaining, altitude, distance_to_goal, velocity_x, velocity_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        
        # ROS Publishers/Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.battery_sub = rospy.Subscriber('/battery', BatteryState, self._battery_cb)
        self.gazebo_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self._gazebo_cb)
        
        # Initialize state variables
        self.battery = 100.0  # %
        self.altitude = 0.0
        self.velocity = np.zeros(2)
        self.goal_position = np.array([10.0, 10.0, 5.0])  # Example goal (x,y,z)
        
        # Energy parameters
        self.energy_consumed = 0.0
        self.base_energy_rate = 0.1  # Energy consumed per second at hover

    def _battery_cb(self, msg):
        self.battery = msg.percentage

    def _gazebo_cb(self, msg):
        try:
            drone_idx = msg.name.index('hector_quadrotor::base_link')
            self.altitude = msg.pose[drone_idx].position.z
            self.velocity = np.array([
                msg.twist[drone_idx].linear.x,
                msg.twist[drone_idx].linear.y
            ])
        except ValueError:
            pass

    def _get_state(self):
        return np.array([
            self.battery,
            self.altitude,
            np.linalg.norm(self.goal_position[:2] - self.velocity),
            self.velocity[0],
            self.velocity[1]
        ])

    def _calculate_reward(self):
        # Reward for reaching goal
        distance_to_goal = np.linalg.norm(self.goal_position[:2] - self.velocity)
        goal_reward = 100.0 if distance_to_goal < 0.5 else -0.1 * distance_to_goal
        
        # Penalize energy consumption
        energy_penalty = -0.5 * self.energy_consumed
        
        # Total reward
        return goal_reward + energy_penalty

    def step(self, action):
        # Execute action
        cmd_vel = Twist()
        cmd_vel.linear.x = action[0]
        cmd_vel.linear.y = action[1]
        cmd_vel.linear.z = action[2]
        cmd_vel.angular.z = action[3]
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Update energy (simplified model)
        self.energy_consumed += self.base_energy_rate * (1 + np.linalg.norm(action))
        
        # Get new state
        state = self._get_state()
        reward = self._calculate_reward()
        done = (self.battery <= 0.0) or (np.linalg.norm(state[1:3]) < 0.5
        
        return state, reward, done, {}

    def reset(self):
        # Reset drone to initial position (requires Gazebo service call)
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_world()
        except rospy.ServiceException:
            pass
        
        # Reset energy and state
        self.energy_consumed = 0.0
        return self._get_state()