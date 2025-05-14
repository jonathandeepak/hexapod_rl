#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import rospy

from hexapod_rl.phantomx_env import PhantomXEnv
from hexapod_rl.reward_plot_headless import RewardPlotHeadlessCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Enable live plotting
plt.ion()

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, model_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.model_path = model_path  # full path with filename

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(self.model_path)
            if self.verbose > 0:
                print(f" Model checkpoint overwritten at {self.num_timesteps} → {self.model_path}")
        return True

def main():
    rospy.init_node("phantomx_rl_train", anonymous=True)
    env = PhantomXEnv()
    check_env(env, warn=True)

    model_name = "phantomx_ppo_model_v1"
    save_dir = os.path.expanduser('~/phantom_ws/src/hexapod_rl/RL_Models/')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, model_name + ".zip")

    if os.path.exists(model_path):
        print(" Loading existing model...")
        model = PPO.load(model_path, env=env)
    else:
        print(" No existing model found, creating new one...")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
        )

    callback = CallbackList([
        RewardPlotHeadlessCallback(save_dir=save_dir),
        CheckpointCallback(save_freq=50000, model_path=model_path, verbose=1)
    ])

    try:
        model.learn(total_timesteps=1000000, callback=callback)
    except (KeyboardInterrupt, SystemExit):
        print("\n Training interrupted by user. Saving model...")

    model.save(model_path)
    print(f" Training complete. Model saved as '{model_path}'")

if __name__ == "__main__":
    main()
