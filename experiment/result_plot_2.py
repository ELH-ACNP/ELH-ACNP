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

class Artist(object):
    data_reshaped: dict
    start: int
    end: int
    step: int

    def __init__(self):
        self.data_reshaped = {}
        self.start = 1
        self.end = 7
        self.step = 1
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
        with open('../data/result_sat_num.pkl', 'rb') as f:
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
        # ax.set_xlim([0, 1500])
        # ax.set_ylim([0.70, 1])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            if key == "alns_cnp":
                continue
            else:
                ax.errorbar(x_values, np.array(self.data_reshaped[key].mean_benefit) /
                            np.array(self.data_reshaped[key].total_benefit), xerr=None,
                            yerr=np.array(self.data_reshaped[key].std_benefit) / np.array(
                                self.data_reshaped[key].total_benefit),
                            linestyle="--",
                            color=self.color_palette[key],
                            linewidth=1,
                            ecolor=self.color_palette[key],
                            elinewidth=1,
                            capsize=5,
                            capthick=2,
                            marker='o',
                            markersize=6,
                            markeredgecolor="black",
                            markeredgewidth=1,
                            markerfacecolor=self.color_palette[key],
                            label=self.new_labels[key]
                            )
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Number of Satellites', fontsize=14, labelpad=2)
        ax.set_ylabel('Benefit Completion Rate', fontsize=14, labelpad=2)
        ax.legend(fontsize=14, ncol=2)
        plt.tight_layout()
        plt.savefig("diff_sat_num_bcr.pdf")
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
        # plt.show()

    def draw_runtime_mix_mean_error_graph(self):
        """
        draw the runtime graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        # ax.set_xlim([0, 1500])
        # ax.set_ylim([0, 40])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
            if key == "alns_cnp":
                continue
            else:
                ax.errorbar(x_values, self.data_reshaped[key].mean_time, xerr=None,
                            yerr=self.data_reshaped[key].std_time,
                            linestyle="--",
                            color=self.color_palette[key],
                            linewidth=1,
                            ecolor=self.color_palette[key],
                            elinewidth=1,
                            capsize=5,
                            capthick=2,
                            marker='o',
                            markersize=6,
                            markeredgecolor="black",
                            markeredgewidth=1,
                            markerfacecolor=self.color_palette[key],
                            label=self.new_labels[key]
                            )
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Number of Satellites', fontsize=14, labelpad=2)
        ax.set_ylabel('Algorithm Running Time(s)', fontsize=14, labelpad=2)
        ax.legend(fontsize=14, ncol=2)
        plt.tight_layout()
        plt.savefig("diff_sat_num_runtime.pdf")
        plt.show()

    def run(self):
        self.read_and_reorganise_data()
        self.draw_benefit_mix_mean_error_graph()
        self.draw_runtime_mix_mean_error_graph()


if __name__ == "__main__":
    artist = Artist()
    artist.run()
