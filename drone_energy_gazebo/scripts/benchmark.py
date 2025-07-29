#!/usr/bin/env python3

import time
import numpy as np
from drone_rl_env import DroneRLEnv
from stable_baselines3 import DQN, SAC

env = DroneRLEnv()
models = {
    "DQN": DQN.load("dqn_drone"),
    "SAC": SAC.load("sac_drone")
}

for name, model in models.items():
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"{name} Total Reward: {total_reward:.2f}")

def run_planner(env, pattern="Lawnmower"):
    print(f"Running {pattern} planner...")
    start = time.time()
    total_energy = 0
    for i in range(5):
        for dx in [0.5, -0.5]:
            action = np.array([dx, 0, 0, 0])
            _, _, _, info = env.step(action)
            total_energy += info['energy_step']
    print(f"{pattern} Time: {time.time()-start:.2f}s, Energy: {total_energy:.2f}")

run_planner(env, "Lawnmower")
