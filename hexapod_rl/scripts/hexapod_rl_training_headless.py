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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


# Enable live plotting
plt.ion()


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_save_path = os.path.join(self.save_path, f"model_step_{self.n_calls}.zip")
            self.model.save(model_save_path)
            if self.verbose > 0:
                print(f"Model checkpoint saved at step {self.n_calls}")
        return True


def main():
    # Initialize ROS node
    rospy.init_node("phantomx_rl_train", anonymous=True)

    # Create the custom environment
    env = PhantomXEnv()

    # Validate the Gym interface
    check_env(env, warn=True)

    # Define model path
    model_path = "phantomx_dqn_model_v7"
    save_dir = os.path.expanduser('~/phantom_ws/plots')
    os.makedirs(save_dir, exist_ok=True)

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

    # Combine callbacks: reward plotting + periodic saving
    callback = CallbackList([
        RewardPlotHeadlessCallback(save_dir=save_dir),
        CheckpointCallback(save_freq=50000, save_path=save_dir, verbose=1)
    ])

    try:
        # Train the model
        model.learn(total_timesteps=1000000, callback=callback)
    except (KeyboardInterrupt, SystemExit):
        print("\n Training interrupted by user. Saving model...")

    # Save final model
    model.save(model_path)
    print(f" Training complete. Model saved as '{model_path}.zip'.")


if __name__ == "__main__":
    main()
