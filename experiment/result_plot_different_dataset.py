import pickle
from dataclasses import dataclass, field
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

# plt.style.use(['science', 'ieee'])

from matplotlib.font_manager import FontProperties
font = FontProperties(family="SimSun", size=14)
plt.style.use(['science', "no-latex"])
@dataclass
class AlgorithmResult(object):
    total_time: list = field(default_factory=list)
    bid_time: list = field(default_factory=list)
    disposal_time: list = field(default_factory=list)

class Painter(object):
    data_reshaped: dict
    start: int
    end: int
    step: int

    def __init__(self):
        self.data_reshaped = {}
        self.start = 50
        self.end = 1501
        self.step = 20
        self.marker_pplette = ["s", "*", "h", "D", "o", "v"]
        self.color_palette = {222: '#E0B300',
                              444: '#E00500',
                              666: '#00E049',
                              888: '#010EE0',
                              114514: '#218F44'
                              }

    def read_and_reorganise_data(self):
        """
        read the running results and reorganize it
        :return:
        """
        with open('../data/result_seed_task.pkl', 'rb') as f:
            results = pickle.load(f)
        navigator = list(results.keys())[0]
        for key in iter(results):
            self.data_reshaped[key] = AlgorithmResult()
        for index in range(len(results[navigator])):
            for key in iter(results):
                self.data_reshaped[key].total_time.append(results[key][index][0])
                self.data_reshaped[key].bid_time.append(results[key][index][1])
                self.data_reshaped[key].disposal_time.append(results[key][index][2])

    def draw_benefit_mix_mean_error_graph(self):
        """
        draw the benefit graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(nrows=3, dpi=200, figsize=(10, 8))
        # ax.set_xlim([0, 1500])
        # ax.set_ylim([0.70, 1])

        x_values = np.arange(self.start, self.end, self.step)
        for index, key in enumerate(self.data_reshaped):
            ax[0].scatter(x_values, np.array(self.data_reshaped[key].total_time)[::4],
                          linestyle="solid",
                          marker=self.marker_pplette[index],
                          edgecolors=self.color_palette[key],
                          facecolors='none',
                          label="dataset{}".format(index),
                          linewidth=1)
            ax[1].scatter(x_values, np.array(self.data_reshaped[key].bid_time)[::4],
                          linestyle="solid",
                          marker=self.marker_pplette[index],
                          edgecolors=self.color_palette[key],
                          facecolors='none',
                          label="dataset{}".format(index),
                          linewidth=1)
            ax[2].scatter(x_values, np.array(self.data_reshaped[key].disposal_time)[::4],
                          linestyle="solid",
                          marker=self.marker_pplette[index],
                          edgecolors=self.color_palette[key],
                          facecolors='none',
                          label="dataset{}".format(index),
                          linewidth=1)
            # y_low = (np.array(self.data_reshaped[key].mean_benefit) - np.array(self.data_reshaped[key].std_benefit))/\
            #         np.array(self.data_reshaped[key].total_benefit)
            # y_high = (np.array(self.data_reshaped[key].mean_benefit) + np.array(self.data_reshaped[key].std_benefit))/\
            #          np.array(self.data_reshaped[key].total_benefit)
            # ax.fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.3)
        ax[0].tick_params(axis='both', which='major', labelsize=16)

        ax[0].set_xlabel('(a)', fontsize=16, labelpad=2)
        # ax[0].set_ylabel('Benefit Completion Rate', fontsize=14, labelpad=2)
        ax[0].legend(fontsize=16, ncol=2)
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[1].set_xlabel('(b)', fontsize=16, labelpad=2)
        ax[1].set_ylabel('算法执行时间(s)', fontproperties=font, fontsize=16, labelpad=2)
        ax[1].legend(fontsize=16, ncol=2)
        ax[2].tick_params(axis='both', which='major', labelsize=16)
        ax[2].set_xlabel('(c)\n任务数量', fontproperties=font, fontsize=16, labelpad=2)
        # ax[2].set_ylabel('Benefit Completion Rate', fontsize=14, labelpad=2)
        ax[2].legend(fontsize=16, ncol=2)
        fig.subplots_adjust(hspace=0.4)
        plt.tight_layout()
        plt.savefig("level_runtime.pdf")
        plt.show()

        # # Add labels and title

        # ax.set_title('Algorithm Convergence Curve', fontsize=8)
        # ax.tick_params(axis='x', labelsize=8)
        # ax.tick_params(axis='y', labelsize=8)
        #
        # # Customize the plot
        # plt.grid(False)
        # # plt.ylim(0, 1)
        #
        # # Show the plot
    #     # plt.show()
    #
    # def draw_runtime_mix_mean_error_graph(self):
    #     """
    #     draw the runtime graph: mean value with error region
    #     :return:
    #     """
    #     fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
    #     ax.set_xlim([50, 1500])
    #     ax.set_ylim([0, 40])
    #     x_values = np.arange(self.start, self.end, self.step)
    #     for key in self.data_reshaped:
    #         ax.plot(x_values, self.data_reshaped[key].mean_time, color=self.color_palette[key],  linestyle="solid",
    #                 label=key, linewidth=1)
    #
    #         y_low = np.array(self.data_reshaped[key].mean_time) - np.array(self.data_reshaped[key].std_time)
    #         y_high = np.array(self.data_reshaped[key].mean_time) + np.array(self.data_reshaped[key].std_time)
    #         ax.fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.3)
    #     ax.tick_params(axis='both', which='major', labelsize=10)
    #     ax.set_xlabel('Number of Tasks', fontsize=10, labelpad=2)
    #     ax.set_ylabel('Algorithm Running Time(s)', fontsize=10, labelpad=2)
    #     ax.legend(fontsize=10, ncol=2)
    #     plt.show()

    def run(self):
        self.read_and_reorganise_data()
        self.draw_benefit_mix_mean_error_graph()
        # self.draw_runtime_mix_mean_error_graph()


if __name__ == "__main__":
    painter = Painter()
    painter.run()
