import pickle
from dataclasses import dataclass, field
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(family="SimSun", size=14)
plt.style.use(['science', "no-latex"])


@dataclass
class AlgorithmResult(object):
    mean_time: list = field(default_factory=list)
    std_time: list = field(default_factory=list)
    mean_benefit: list = field(default_factory=list)
    std_benefit: list = field(default_factory=list)
    total_benefit: list = field(default_factory=list)

class Painter(object):
    data_reshaped: dict
    start: int
    end: int
    step: int

    def __init__(self):
        self.data_reshaped = {}
        self.start = 0
        self.end = 101
        self.step = 1
        self.color_palette = {"pre_tuned_alns_cnp": 'green',  # BORDEAUX
                              3: '#800020',  # BURGUNDY
                              4: '#B05923',  # CHINA RED
                              5: '#002FA7',  # KLEIN BLUE
                              6: '#003153',  # PRUSSIAN BLUE
                              "ns_cnp": '#81D8D0',  # TIFFANY BLUE
                              "hpfs_cnp": '#008C8C',  # MARS GREEN
                              "fcfs_cnp": '#F9DC24',  # SENNELIER YELLOW
                              "llf_cnp": '#E85827'  # HERMES ORANGE
                              }

    def read_and_reorganise_data(self):
        """
        read the running results and reorganize it
        :return:
        """
        with open('../data/result_convergence.pkl', 'rb') as f:
            self.data_reshaped = pickle.load(f)

    def draw_benefit_mix_mean_error_graph(self):
        """
        draw the benefit graph: mean value with error region
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
        ax.set_xlim([-0.5, 100])
        # ax.set_ylim([0.75, 1])
        x_values = np.arange(self.start, self.end, self.step)
        for key in self.data_reshaped:
                ax.plot(x_values, np.array(self.data_reshaped[key]), color=self.color_palette[key], linestyle="solid", marker="*",
                        label=f"{key} DRSs", linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel('Negotiation round', fontsize=14, labelpad=2)
        ax.set_ylabel('Benefit Completion Rate', fontsize=14, labelpad=2)
        ax.legend(fontsize=14, ncol=2)
        plt.savefig("convergence.pdf")
        plt.show()

    def run(self):
        self.read_and_reorganise_data()
        self.draw_benefit_mix_mean_error_graph()


if __name__ == "__main__":
    painter = Painter()
    painter.run()
