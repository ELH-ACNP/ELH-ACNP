import pickle
from dataclasses import dataclass, field
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(['science', 'ieee'])

# from matplotlib.font_manager import FontProperties
# font = FontProperties(family="SimSun", size=14)
# plt.style.use(['science', "no-latex"])

@dataclass
class AlgorithmResult(object):
    mean_time: list = field(default_factory=list)
    std_time: list = field(default_factory=list)
    mean_benefit: list = field(default_factory=list)
    std_benefit: list = field(default_factory=list)
    total_benefit: list = field(default_factory=list)

class Illustrator(object):
    data_reshaped: dict
    start: float
    end: float
    step: float

    def __init__(self):
        self.data_reshaped = {}
        self.start = 0.01
        self.end = 0.51
        self.step = 0.01
        self.color_palette = {"pre_tuned_alns_cnp": '#FA0B00',  # BORDEAUX
                              "alns_cnp": '#0100FA',  # BURGUNDY
                              "lns_cnp": '#FACE05',  # CHINA RED
                              "vns_cnp": '#05FA62',  # KLEIN BLUE
                              "vnd_cnp": '#A5923A',  # PRUSSIAN BLUE
                              "ns_cnp": '#7A4340',  # TIFFANY BLUE
                              "hpfs_cnp": '#40407A',  # MARS GREEN
                              "fcfs_cnp": '#407A56',  # SENNELIER YELLOW
                              "llf_cnp": '#73EA8B'  # HERMES ORANGE
                              }
        self.new_labels = {"pre_tuned_alns_cnp": 'ELH-ACNP',
                           "alns_cnp": 'EH-ACNP',
                           "lns_cnp": 'LNS-CNP',
                           "vns_cnp": 'VNS-CNP',
                           "vnd_cnp": 'VND-CNP',
                           "ns_cnp": 'H-CNP',
                           "hpfs_cnp": 'HPFS-CNP',
                           "fcfs_cnp": 'FCFS-CNP',
                           "llf_cnp": 'LLF-CNP'
                           }

    def read_and_reorganise_data(self):
        """
        read the running results and reorganize it
        :return:
        """
        with open('../data/result_differ_gama.pkl', 'rb') as f:
            results = pickle.load(f)
        navigator = list(results.keys())[0]
        for key in iter(results):
            self.data_reshaped[key] = AlgorithmResult()
        for index in range(len(results[navigator])):
            for key in iter(results):
                self.data_reshaped[key].mean_time.append(results[key][index][0])
                self.data_reshaped[key].std_time.append(results[key][index][1])
                self.data_reshaped[key].mean_benefit.append(results[key][index][2])
                self.data_reshaped[key].std_benefit.append(results[key][index][3])
                self.data_reshaped[key].total_benefit.append(results[key][index][4])

    def draw_benefit_mix_mean_error_graph(self):
        """
        draw the benefit graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        ax.set_xlim([0.01, 0.5])
        # ax.set_ylim([0.70, 1])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            if key == "alns_cnp":
                continue
            else:
                ax.plot(x_values, np.array(self.data_reshaped[key].mean_benefit) /
                        np.array(self.data_reshaped[key].total_benefit),
                        color=self.color_palette[key],
                        linestyle="solid",
                        label=self.new_labels[key],
                        linewidth=1)
                y_low = (np.array(self.data_reshaped[key].mean_benefit) - np.array(self.data_reshaped[key].std_benefit))/\
                        np.array(self.data_reshaped[key].total_benefit)
                y_high = (np.array(self.data_reshaped[key].mean_benefit) + np.array(self.data_reshaped[key].std_benefit))/\
                         np.array(self.data_reshaped[key].total_benefit)
                ax.fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.35)
        ax.axvline(x=0.24, color='b', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('任务抛弃比例' r'$\rho$' '取值', fontproperties=font, fontsize=14, labelpad=2)
        ax.set_ylabel('收益完成率', fontproperties=font, fontsize=14, labelpad=2)
        ax.legend(fontsize=14, ncol=2)
        plt.savefig("disposal_rate_bcr.pdf")
        plt.show()

    def draw_runtime_mix_mean_error_graph(self):
        """
        draw the runtime graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        ax.set_xlim([0.01, 0.5])
        # ax.set_ylim([0, 40])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            if key == "alns_cnp":
                continue
            else:
                ax.plot(x_values, self.data_reshaped[key].mean_time,
                        color=self.color_palette[key],
                        linestyle="solid",
                        label=self.new_labels[key],
                        linewidth=1)

                y_low = np.array(self.data_reshaped[key].mean_time) - np.array(self.data_reshaped[key].std_time)
                y_high = np.array(self.data_reshaped[key].mean_time) + np.array(self.data_reshaped[key].std_time)
                ax.fill_between(x_values, y_low, y_high, color=self.color_palette[key], alpha=0.35)
        plt.axvline(x=0.18, color='b', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('任务抛弃比例' r'$\rho$' '取值', fontproperties=font, fontsize=14, labelpad=2)
        ax.set_ylabel('算法执行时间(s)', fontproperties=font, fontsize=14, labelpad=2)
        ax.legend(fontsize=14, ncol=2)
        plt.savefig("disposal_runtime.pdf")
        plt.show()

    def run(self):
        self.read_and_reorganise_data()
        self.draw_benefit_mix_mean_error_graph()
        self.draw_runtime_mix_mean_error_graph()


if __name__ == "__main__":
    illustrator = Illustrator()
    illustrator.run()





