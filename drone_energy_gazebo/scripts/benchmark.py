import numpy as np
from drone_rl_env import DroneRLEnv
from stable_baselines3 import SAC, DQN

def run_planner(env, planner_type="A*"):
    if planner_type == "A*":
        # Implement A* path planning
        pass
    elif planner_type == "Lawnmower":
        # Implement lawnmower pattern
        pass

# Compare RL vs. Traditional Planners
env = DroneRLEnv()
rl_models = {
    "DQN": DQN.load("dqn_drone"),
    "SAC": SAC.load("sac_drone")
}

for name, model in rl_models.items():
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"{name} Total Reward: {total_reward}")

# Run traditional planners
run_planner(env, "A*")
run_planner(env, "Lawnmower")