#!/usr/bin/env python3


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
    model_path = os.path.expanduser('~/phantom_ws/src/hexapod_rl/RL_Models/phantomx_ppo_model_v1.zip')
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
