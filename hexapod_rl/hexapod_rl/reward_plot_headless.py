#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class RewardPlotHeadlessCallback(BaseCallback):
    def __init__(self, save_dir='~/phantom_ws/src/hexapod_rl/plots/training/', filename='reward_plot.png', verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.rolling_rewards = []
        self.save_dir = save_dir
        self.filename = filename

        os.makedirs(self.save_dir, exist_ok=True)
        self.fig, self.ax = plt.subplots()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        if dones and dones[0]:  # End of episode
            episode_reward = sum(rewards)
            self.episode_rewards.append(episode_reward)

            # Calculate rolling average (last 10 episodes)
            if len(self.episode_rewards) >= 10:
                avg_reward = sum(self.episode_rewards[-10:]) / 10
            else:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)

            self.rolling_rewards.append(avg_reward)

            # Update the plot
            self.ax.clear()
            self.ax.plot(self.episode_rewards, label="Episode Reward")
            self.ax.plot(self.rolling_rewards, label="Rolling Avg (10)")
            self.ax.set_title("Episode Rewards Over Time")
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")
            self.ax.legend()

            # Overwrite the same image file each time
            out_path = os.path.join(self.save_dir, self.filename)
            self.fig.savefig(out_path, bbox_inches='tight')

        return True

    def _on_training_end(self) -> None:
        # Optionally save one final version
        final_path = os.path.join(self.save_dir, "reward_plot_final.png")
        self.fig.savefig(final_path, bbox_inches='tight')
