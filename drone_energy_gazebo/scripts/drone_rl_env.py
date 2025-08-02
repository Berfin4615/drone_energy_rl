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
from geometry_msgs.msg import Twist, Quaternion
from hector_uav_msgs.msg import Supply
from std_srvs.srv import Empty
import tf.transformations as tf_trans

class DroneRLEnv(gym.Env):
    def __init__(self):
        super(DroneRLEnv, self).__init__()
        
        # Initialize battery parameters
        self.full_voltage = 14.8
        self.empty_voltage = 10.0
        self.battery_voltage = self.full_voltage
        self.battery_percentage = 100.0
        
        # Initialize ROS components
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/supply', Supply, self._supply_cb)
        rospy.Subscriber('/gazebo/link_states', LinkStates, self._gazebo_cb)
        
        # Wait for publishers to connect
        rospy.loginfo("Waiting for cmd_vel publisher connections...")
        while self.cmd_vel_pub.get_num_connections() < 1 and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Publisher connected!")
        
        # Environment parameters
        self.max_steps_per_episode = 200
        self.flip_threshold = 0.7854  # 45 degrees in radians
        
        # Environment boundaries - 10x10 meter area
        self.x_min, self.x_max = -10.0, 10.0
        self.y_min, self.y_max = -10.0, 10.0
        self.z_min, self.z_max = 0.1, 10.0
        
        # Goal parameters - within boundaries
        self.goal_position = np.array([8.0, 8.0, 0.55])
        self.goal_radius = 1.0
        self.landing_threshold = 0.3
        
        # Action space (9 discrete actions)
        self.action_space = spaces.Discrete(9)
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
            high=np.array([100, 100, 20, 10, 10, 10]),
            dtype=np.float32
        )

        # Initialize state variables
        self.current_position = np.array([-9.0, -9.0, 1.0])
        self.velocity = np.zeros(2)
        self.orientation = Quaternion()
        self.step_count = 0
        
        # Initialize previous distance
        horizontal_dist = np.linalg.norm(self.goal_position[:2] - self.current_position[:2])
        vertical_dist = abs(self.goal_position[2] - self.current_position[2])
        self.prev_dist = horizontal_dist + vertical_dist
        
        rospy.loginfo("Environment initialized!")
        rospy.sleep(1.0)  # Allow time for initialization
    def _supply_cb(self, msg):
        if len(msg.voltage) > 0:
            self.battery_voltage = msg.voltage[0]
            self.battery_percentage = max(0.0, min(100.0, 
                100 * (self.battery_voltage - self.empty_voltage) / 
                (self.full_voltage - self.empty_voltage)
            ))

    def _gazebo_cb(self, msg):
        try:
            # Find the index for the drone's base link
            idx = -1
            for i, name in enumerate(msg.name):
                if 'quadrotor::base_link' in name:
                    idx = i
                    break
            
            if idx == -1:
                rospy.logwarn("Drone base_link not found in Gazebo link states!")
                return
                
            pos = msg.pose[idx].position
            orient = msg.pose[idx].orientation
            vel = msg.twist[idx].linear
            
            self.current_position = np.array([pos.x, pos.y, pos.z])
            self.velocity = np.array([vel.x, vel.y])
            self.orientation = orient
            
            rospy.logdebug(f"Position updated: {self.current_position}")
        except Exception as e:
            rospy.logerr(f"Error in Gazebo callback: {str(e)}")

    def _is_flipped(self):
        """Check if drone has flipped over"""
        try:
            quaternion = [
                self.orientation.x,
                self.orientation.y,
                self.orientation.z,
                self.orientation.w
            ]
            roll, pitch, _ = tf_trans.euler_from_quaternion(quaternion)
            
            if abs(roll) > self.flip_threshold or abs(pitch) > self.flip_threshold:
                rospy.logwarn(f"Drone flipped! Roll: {roll:.2f} rad, Pitch: {pitch:.2f} rad")
                return True
        except Exception as e:
            rospy.logerr(f"Flip detection error: {str(e)}")
        return False

    def _get_state(self):
        horizontal_dist = np.linalg.norm(self.goal_position[:2] - self.current_position[:2])
        vertical_dist = abs(self.goal_position[2] - self.current_position[2])
        return np.array([
            self.battery_percentage,
            self.current_position[2],
            horizontal_dist,
            vertical_dist,
            self.velocity[0],
            self.velocity[1]
        ], dtype=np.float32)

    def _calculate_distance_reward(self, horizontal_dist, vertical_dist):
        """Calculate the core reward based on distance to goal"""
        # Calculate current total distance to goal
        current_dist = horizontal_dist + vertical_dist
        
        # Calculate improvement from previous step
        improvement = self.prev_dist - current_dist
        self.prev_dist = current_dist
        
        # Base reward is proportional to improvement
        reward = 20.0 * improvement  # Strong reward for moving closer
        
        # Additional reward for being close to goal
        if horizontal_dist < self.goal_radius and vertical_dist < 1.0:
            reward += 2.0 * (1.0 - horizontal_dist/self.goal_radius)
        
        return reward

    def _is_out_of_bounds(self):
        x, y, z = self.current_position
        out = (x < self.x_min or x > self.x_max or 
               y < self.y_min or y > self.y_max or 
               z < self.z_min or z > self.z_max)
        if out:
            rospy.logwarn(f"OUT OF BOUNDS! Position: [{x:.2f}, {y:.2f}, {z:.2f}] "
                          f"Bounds: X[{self.x_min}-{self.x_max}] "
                          f"Y[{self.y_min}-{self.y_max}] "
                          f"Z[{self.z_min}-{self.z_max}]")
        return out

    def step(self, action_idx):
        # Increment step counter
        self.step_count += 1
        
        # Execute action
        action = self.action_mapping[action_idx]
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z, cmd.angular.z = action
        
        rospy.loginfo(f"Executing action {action_idx}: {action}")
        self.cmd_vel_pub.publish(cmd)
        
        # Wait for simulation step
        rospy.sleep(0.1)
        
        # Force a position update
        try:
            rospy.loginfo("Waiting for position update...")
            self._gazebo_cb(rospy.wait_for_message('/gazebo/link_states', LinkStates, timeout=1.0))
        except rospy.ROSException:
            rospy.logwarn("Timed out waiting for position update!")
        
        # Get new state
        state = self._get_state()
        horizontal_dist = state[2]
        vertical_dist = state[3]
        
        # Check termination conditions
        out_of_bounds = self._is_out_of_bounds()
        landed = (horizontal_dist < self.goal_radius) and (vertical_dist < self.landing_threshold)
        battery_dead = self.battery_percentage <= 0
        flipped = self._is_flipped()

        terminated = landed or battery_dead or out_of_bounds or flipped
        truncated = False
        
        # Check if max steps reached
        if not terminated and self.step_count >= self.max_steps_per_episode:
            rospy.loginfo("Max steps reached! Resetting environment.")
            truncated = True
        
        # Calculate core reward based on distance improvement
        distance_reward = self._calculate_distance_reward(horizontal_dist, vertical_dist)
        
        # Apply terminal rewards and penalties
        if landed:
            reward = 200.0 + distance_reward
            rospy.loginfo("SUCCESS! Drone landed at goal position!")
        elif battery_dead:
            reward = -100.0
            rospy.loginfo("FAILURE! Battery depleted!")
        elif out_of_bounds:
            reward = -50.0
            rospy.loginfo("FAILURE! Drone went out of bounds!")
        elif flipped:
            reward = -50.0
            rospy.loginfo("FAILURE! Drone flipped over!")
        else:
            # Normal step reward
            reward = distance_reward - 0.1
        
        # Log state information
        rospy.loginfo(f"Position: {self.current_position} | "
                      f"Goal: {self.goal_position} | "
                      f"Dist: H{horizontal_dist:.2f} V{vertical_dist:.2f} | "
                      f"Steps: {self.step_count}/{self.max_steps_per_episode} | "
                      f"Reward: {reward:.2f} | "
                      f"Terminated: {terminated} | "
                      f"Truncated: {truncated}")
        
        info = {
            "battery": self.battery_percentage,
            "position": self.current_position,
            "distance_to_goal": horizontal_dist,
            "altitude_diff": vertical_dist,
            "landed": landed,
            "out_of_bounds": out_of_bounds,
            "flipped": flipped,
            "step_count": self.step_count
        }
        
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        rospy.loginfo("Resetting environment...")
        
        # Reset simulation
        try:
            rospy.wait_for_service('/gazebo/reset_world', timeout=5.0)
            reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_response = reset_world()
            rospy.loginfo("Gazebo world reset!")
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"Reset failed: {e}")
        
        # Reset state variables
        self.battery_voltage = self.full_voltage
        self.battery_percentage = 100.0
        self.step_count = 0
        
        # Set initial position
        self.current_position = np.array([-9.0, -9.0, 1.0])
        
        # Re-initialize previous distance
        horizontal_dist = np.linalg.norm(self.goal_position[:2] - self.current_position[:2])
        vertical_dist = abs(self.goal_position[2] - self.current_position[2])
        self.prev_dist = horizontal_dist + vertical_dist
        
        rospy.sleep(1.0)  # Allow time for reset
        
        # Force position update
        try:
            rospy.loginfo("Waiting for initial position update...")
            self._gazebo_cb(rospy.wait_for_message('/gazebo/link_states', LinkStates, timeout=2.0))
            rospy.loginfo(f"Initial position: {self.current_position}")
        except rospy.ROSException:
            rospy.logwarn("Timed out waiting for initial position update!")
        
        return self._get_state(), {}

    def close(self):
        rospy.loginfo("Closing DroneRLEnv")
        self.cmd_vel_pub.unregister()