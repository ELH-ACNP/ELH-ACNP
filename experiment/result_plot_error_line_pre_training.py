import pickle
from dataclasses import dataclass, field
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee'])

from matplotlib.font_manager import FontProperties
font = FontProperties(family="SimSun", size=14)
plt.style.use(['science', "no-latex"])

class Painter(object):
    error_curve: list

    def __init__(self):
        pass

    def read_data(self):
        with open('../data/weight_trained_curve.pkl', 'rb') as f:
            self.error_curve = pickle.load(f)

    def plot_line(self):
        """
        plot error line
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        x_values = np.arange(1, len(self.error_curve) + 1)
        ax.set_xlim([1, max(2, len(self.error_curve))])
        ax.set_ylim(bottom=0)
        ax.plot(x_values, self.error_curve, color='#FA0B00', linestyle="solid", linewidth=2, marker="*")
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Training update', fontsize=14, labelpad=2)
        ax.set_ylabel('Chebyshev error', fontsize=14, labelpad=2)
        plt.savefig("train_error_curve.pdf")
        plt.tight_layout()
        plt.show()

    def run(self):
        self.read_data()
        self.plot_line()


if __name__ == "__main__":
    painter = Painter()
    painter.run()


