# #!/usr/bin/env python3

# import rospy
# import csv
# from stable_baselines3 import DQN
# from drone_gym_wrapper import DroneGymWrapper  # wrapper'ı kullanıyoruz

# def main():
#     rospy.init_node('train_dqn')

#     env = DroneGymWrapper()

#     model = DQN(
#         policy='MlpPolicy',
#         env=env,
#         verbose=1,
#         tensorboard_log='./logs/dqn/'
#     )

#     model.learn(total_timesteps=100000)
#     model.save("dqn_drone")

#     with open('training_dqn_log.csv', 'w', newline='') as log_file:
#         writer = csv.writer(log_file)
#         writer.writerow(['Episode', 'Reward', 'Battery%', 'Energy', 'Distance'])

#         for ep in range(100):
#             obs = env.reset()
#             total_reward = 0
#             done = False

#             while not done and not rospy.is_shutdown():
#                 action, _ = model.predict(obs)
#                 obs, reward, done, info = env.step(action)
#                 total_reward += reward

#             writer.writerow([
#                 ep,
#                 round(total_reward, 3),
#                 round(info.get('battery', 0.0), 2),
#                 round(info.get('total_energy', 0.0), 3),
#                 round(info.get('distance_to_goal', 0.0), 3)
#             ])

#     rospy.loginfo("DQN training and evaluation finished.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import rospy
import gymnasium as gym
from stable_baselines3 import DQN
from drone_rl_env import DroneRLEnv

def main():
    rospy.init_node('train_dqn', anonymous=True)
    
    # Create environment
    env = DroneRLEnv()
    
    # Initialize DQN with discrete action space
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        device="auto"
    )
    
    try:
        model.learn(total_timesteps=100000)
        model.save("dqn_drone_model")
        rospy.loginfo("Training completed successfully!")
    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted by user")
    finally:
        env.close()

if __name__ == '__main__':
    main()