from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class RewardPlotCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.rolling_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") and self.locals["dones"][0]:
            ep_reward = sum(self.locals["rewards"])
            self.episode_rewards.append(ep_reward)

            # Rolling average
            if len(self.episode_rewards) >= 10:
                avg = sum(self.episode_rewards[-10:]) / 10
            else:
                avg = sum(self.episode_rewards) / len(self.episode_rewards)
            self.rolling_rewards.append(avg)

            # Plot live
            plt.clf()
            plt.title("Episode Reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.plot(self.episode_rewards, label="Episode")
            plt.plot(self.rolling_rewards, label="Rolling Avg (10)")
            plt.legend()
            plt.pause(0.01)

        return True
