#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import rospy

from hexapod_rl.phantomx_env import PhantomXEnv
from hexapod_rl.reward_plot_headless import RewardPlotHeadlessCallback
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


# Enable live plotting
plt.ion()


def main():
    # Initialize ROS node
    rospy.init_node("phantomx_rl_train", anonymous=True)

    # Create the custom environment
    env = PhantomXEnv()

    # Optional: validate the Gym interface
    check_env(env, warn=True)

    # Define model path
    model_path = "phantomx_dqn_model_v5"

    # Load existing model or create new
    if os.path.exists(model_path + ".zip"):
        print(" Loading existing model...")
        model = DQN.load(model_path, env=env)
    else:
        print(" No existing model found, creating new one...")
        model = DQN(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=0.05,
            gamma=0.99,
            train_freq=4,
            target_update_interval=100,
            verbose=1,
        )

    # Create reward plot callback
    callback = RewardPlotHeadlessCallback(save_dir=os.path.expanduser('~/phantom_ws/plots'))

    try:
        # Train the model
        model.learn(total_timesteps=10000000000, callback=callback)
    except (KeyboardInterrupt, SystemExit):
        print("\n Training interrupted by user. Saving model...")

    # Save model
    model.save(model_path)
    print(" Training complete. Model saved as 'phantomx_dqn_model'.")


if __name__ == "__main__":
    main()
