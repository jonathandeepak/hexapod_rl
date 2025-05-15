import matplotlib.pyplot as plt

class LossLogger:
    def __init__(self):
        self.losses = []

    def log(self, loss_value):
        self.losses.append(loss_value)

    def plot(self, save_path=None):
        plt.figure()
        plt.plot(self.losses, label="Loss")
        plt.title("DQN Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
