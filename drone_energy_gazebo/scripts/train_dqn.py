#!/usr/bin/env python3
import rospy
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_rl_env import DroneRLEnv
import os

def main():
    # Initialize ROS node
    rospy.init_node('drone_rl_training', anonymous=True)
    
    # Create logs directory
    os.makedirs("./dqn_logs/", exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([lambda: DroneRLEnv()])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./dqn_logs/",
        name_prefix="drone_model"
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./dqn_logs/best/",
        log_path="./dqn_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Model parameters
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.99,
        tau=1.0,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log="./dqn_logs/",
        device="auto"
    )
    
    try:
        rospy.loginfo("Starting training... Press Ctrl+C to stop.")
        model.learn(
            total_timesteps=500000,
            callback=[checkpoint_callback, eval_callback],
            log_interval=100,
            tb_log_name="DQN"
        )
        model.save("dqn_drone_final_model")
        rospy.loginfo("Training completed successfully!")
    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted by user")
    except Exception as e:
        rospy.logerr(f"Training failed: {str(e)}")
    finally:
        env.close()

if __name__ == '__main__':
    main()