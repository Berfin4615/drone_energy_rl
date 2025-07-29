#!/usr/bin/env python3
from gym import Env, spaces
import numpy as np
from drone_rl_env import DroneRLEnv

class DroneGymWrapper(Env):
    def __init__(self):
        super().__init__()
        self.env = DroneRLEnv()

        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -5, -5]),
                                            high=np.array([100, 20, 50, 5, 5]),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def step(self, action):
        action_map = {
            0: [0.0, 0.0, 0.0, 0.0],
            1: [0.5, 0.0, 0.0, 0.0],
            2: [0.0, 0.5, 0.0, 0.0],
            3: [0.0, 0.0, 0.5, 0.0],
            4: [0.0, 0.0, 0.0, 0.5],
        }
        continuous_action = action_map[action]
        return self.env.step(continuous_action)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.env.close()
