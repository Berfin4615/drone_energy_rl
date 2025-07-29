import torch
import torch.nn as nn
from drone_rl_env import DroneRLEnv
from stable_baselines3 import DQN

env = DroneRLEnv()
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./logs/dqn/")
model.learn(total_timesteps=100000)

# Save model
model.save("dqn_drone")