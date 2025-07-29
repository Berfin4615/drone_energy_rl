#!/usr/bin/env python3
import rospy
from drone_rl_env import DroneRLEnv
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

rospy.init_node('sac_trainer')

env = Monitor(DroneRLEnv())

# SAC-specific callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path='./logs/sac/',
    name_prefix='sac_model'
)

eval_callback = EvalCallback(
    env,
    best_model_save_path='./logs/sac/best/',
    log_path='./logs/sac/',
    eval_freq=5000,
    deterministic=True,
    render=False
)

model = SAC(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=0.0003,
    buffer_size=300000,
    learning_starts=10000,
    batch_size=256,
    ent_coef='auto',
    gamma=0.99,
    tau=0.005,
    tensorboard_log='./logs/sac/'
)

model.learn(
    total_timesteps=200000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name='sac_run'
)

model.save("sac_energy_drone")
print("SAC training completed.")