import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import rospy

from hexapod_rl.phantomx_env import PhantomXEnv
from hexapod_rl.reward_plot import RewardPlotCallback
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CallbackList


# Enable live plotting
plt.ion()


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"model_step_{self.num_timesteps}.zip")
            self.model.save(checkpoint_file)
            if self.verbose:
                print(f" Saved model checkpoint at step {self.num_timesteps}")
        return True


def main():
    rospy.init_node("phantomx_rl_train", anonymous=True)
    env = PhantomXEnv()
    check_env(env, warn=True)

    model_path = "phantomx_dqn_model_v7"
    checkpoint_dir = "checkpoints"

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

    # Create callbacks
    reward_callback = RewardPlotCallback()
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=checkpoint_dir)
    callback = CallbackList([reward_callback, checkpoint_callback])

    try:
        model.learn(total_timesteps=1000000, callback=callback)
    except (KeyboardInterrupt, SystemExit):
        print("\nTraining interrupted by user. Saving model...")

    model.save(model_path)
    print("Training complete. Model saved as 'phantomx_dqn_model'.")
    

if __name__ == "__main__":
    main()
