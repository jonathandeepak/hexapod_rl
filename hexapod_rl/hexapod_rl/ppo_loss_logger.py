#!/usr/bin/env python3

import matplotlib.pyplot as plt
import os

class PPOLossLogger:
    def __init__(self, live_plot=True, save_path=None):
        self.losses = []
        self.live_plot = live_plot
        self.save_path = save_path

        if self.live_plot:
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.line, = self.ax.plot([], [], label="Loss")
            self.ax.set_title("Training Loss (Live)")
            self.ax.set_xlabel("Training Step")
            self.ax.set_ylabel("Loss")
            self.ax.grid(True)
            self.ax.legend()
            self.fig.show()

    def log(self, loss_value):
        self.losses.append(loss_value)
        if self.live_plot:
            self.line.set_data(range(len(self.losses)), self.losses)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            # Save the current live figure if save_path is set
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                self.fig.savefig(self.save_path)

    def plot(self, save_path=None):
        if self.live_plot:
            plt.ioff()

        final_save_path = save_path or self.save_path
        plt.figure()
        plt.plot(self.losses, label="Loss")
        plt.title("Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        if final_save_path:
            os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
            plt.savefig(final_save_path)
        else:
            plt.show()
