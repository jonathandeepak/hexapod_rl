#!/usr/bin/env python3

# # import os
# # import sys
# # import time
# # import rospy
# # import numpy as np

# # # Add parent directory to import path if necessary
# # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# # from hexapod_rl.phantomx_env import PhantomXEnv
# # from stable_baselines3 import DQN


# # def main():
# #     rospy.init_node("phantomx_inference_node", anonymous=True)

# #     # Create environment
# #     env = PhantomXEnv()

# #     # Load the trained model
# #     model_path = os.path.expanduser('~/phantom_ws/phantomx_dqn_model_v5.zip')
# #     model = DQN.load(model_path, env=env)

# #     obs, _ = env.reset()

# #     rate = rospy.Rate(10)  # 10 Hz control rate
# #     while not rospy.is_shutdown():
# #         action, _states = model.predict(obs, deterministic=True)
# #         print(f"Predicted action: {action}")

# #         obs, reward, terminated, truncated, info = env.step(action)
# #         done = terminated or truncated

# #         rospy.loginfo(f"Reward: {reward:.3f}")

# #         if done:
# #             rospy.loginfo("Episode finished. Resetting environment.")
# #             obs, _ = env.reset()

# #         rate.sleep()

# # if __name__ == "__main__":
# #     main()

# import os
# import sys
# import rospy
# import numpy as np
# from std_srvs.srv import Empty
# # import torch

# # Add parent directory to import path if necessary
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from hexapod_rl.phantomx_env import PhantomXEnv
# from stable_baselines3 import DQN


# def main():
#     rospy.init_node("phantomx_inference_node", anonymous=True)

#     # Unpause Gazebo physics so simulation advances
#     rospy.wait_for_service('/gazebo/unpause_physics')
#     try:
#         unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
#         unpause_sim()
#         rospy.loginfo("Unpaused Gazebo physics.")
#     except rospy.ServiceException as e:
#         rospy.logerr(f"Failed to unpause Gazebo: {e}")

#     # Create environment
#     env = PhantomXEnv()

#     # Load the trained model
#     model_path = os.path.expanduser('~/phantom_ws/src/hexapod_rl/RL_Models/phantomx_dqn_model_v8.zip')
#     model = DQN.load(model_path, env=env)

#     # Reset environment (returns observation and info dict)
#     obs, _ = env.reset()

#     rate = rospy.Rate(10)  # 10 Hz control rate
#     while not rospy.is_shutdown():
#         # Debug: print observation vector
#         if isinstance(obs, dict) and 'vector' in obs:
#             rospy.loginfo(f"Obs vector: {obs['vector']}")
#             # Debug: print Q-values
#             # q_vals = model.q_net(torch.tensor(obs['vector'], dtype=torch.float32).unsqueeze(0))
#             # rospy.loginfo(f"Q-values: {q_vals.detach().numpy()}")
#         else:
#             rospy.loginfo(f"Obs: {obs}")

#         # Predict action (greedy)
#         action, _states = model.predict(obs, deterministic=False)
#         rospy.loginfo(f"Sampled action (ε-greedy): {action}")

#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         rospy.loginfo(f"Reward: {reward:.3f}")

#         # Check for end of episode
#         if done:
#             rospy.loginfo("Episode finished. Resetting environment.")
#             obs, _ = env.reset()

#         rate.sleep()


# if __name__ == "__main__":
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass


import os
import sys
import rospy
import numpy as np
from std_srvs.srv import Empty

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hexapod_rl.phantomx_env import PhantomXEnv
from hexapod_rl.inference_reward_action_plotter import ActionPlotter
from stable_baselines3 import DQN


def main():
    rospy.init_node("phantomx_inference_node", anonymous=True)

    rospy.wait_for_service('/gazebo/unpause_physics')
    try:
        unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        unpause_sim()
        rospy.loginfo("Unpaused Gazebo physics.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Failed to unpause Gazebo: {e}")

    env = PhantomXEnv()
    model_path = os.path.expanduser('~/phantom_ws/src/hexapod_rl/RL_Models/phantomx_dqn_model_v6.zip')
    model = DQN.load(model_path, env=env)

    obs, _ = env.reset()
    rate = rospy.Rate(10)

    # Initialize action plotter
    logger = ActionPlotter()

    while not rospy.is_shutdown():
        action, _ = model.predict(obs, deterministic=False)
        logger.log(action)

        rospy.loginfo(f"Sampled action: {action}")
        obs, reward, terminated, truncated, _ = env.step(action)
        rospy.loginfo(f"Reward: {reward:.3f}")

        if terminated or truncated:
            rospy.loginfo("Episode finished. Resetting environment.")
            obs, _ = env.reset()

        rate.sleep()

    # Save plot when ROS shuts down
    logger.plot()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
