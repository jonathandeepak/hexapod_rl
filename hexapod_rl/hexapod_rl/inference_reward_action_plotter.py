#!/usr/bin/env python3
import matplotlib.pyplot as plt
from collections import Counter

class ActionPlotter:
    def __init__(self):
        self.counter = Counter()

    def log(self, action):
        """Log an action (cast to int for compatibility)."""
        self.counter[int(action)] += 1

    def plot(self, save_path="~/phantom_ws/src/hexapod_rl/plots/inference/"):
        """Plot a histogram of the logged actions."""
        actions = list(self.counter.keys())
        counts = list(self.counter.values())

        plt.figure(figsize=(8, 6))
        plt.bar(actions, counts)
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Histogram of Predicted Actions')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
