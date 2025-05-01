#!/usr/bin/env python3
import os
import matplotlib
# switch to non-interactive backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

class  RewardPlotHeadlessCallback(BaseCallback):
    def __init__(self, save_dir='plots'):
        super().__init__()
        self.episode_rewards = []
        self.rolling_rewards = []

        # where to dump PNGs
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # one figure & axes for all callbacks
        self.fig, self.ax = plt.subplots()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        rewards = self.locals.get("rewards", [])

        # only on episode end
        if dones and dones[0]:
            # accumulate the just-finished episode's reward
            ep_reward = sum(rewards)
            self.episode_rewards.append(ep_reward)

            # compute rolling average over last 10
            if len(self.episode_rewards) >= 10:
                avg = sum(self.episode_rewards[-10:]) / 10
            else:
                avg = sum(self.episode_rewards) / len(self.episode_rewards)
            self.rolling_rewards.append(avg)

            # plot
            self.ax.clear()
            self.ax.plot(self.episode_rewards, label="Episode Reward")
            self.ax.plot(self.rolling_rewards, label="Rolling Avg (10)")
            self.ax.set_title("Episode Reward")
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Reward")
            self.ax.legend()

            # save to file
            ep_num = len(self.episode_rewards)
            out_path = os.path.join(self.save_dir, f"reward_plot_{ep_num:03d}.png")
            self.fig.savefig(out_path, bbox_inches='tight')
            # no plt.pause() needed in headless

        return True
