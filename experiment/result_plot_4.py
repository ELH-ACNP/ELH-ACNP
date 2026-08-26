import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import pickle

# plt.style.use(['science', 'ieee'])

from matplotlib.font_manager import FontProperties
font = FontProperties(family="SimSun", size=14)
plt.style.use(['science', "no-latex"])

class Drawer(object):
    trained_weight: np.ndarray
    destroy_operator: list
    repair_operator: list

    def __init__(self):
        self.destroy_operator = ["随机弃标", "最小权重优先弃标", "最大冲突优先弃标",
                                 "最低效率优先弃标"]
        self.repair_operator = ["随机招标", "最大奖励优先招标", "最小冲突优先招标",
                                "效率优先招标"]

    def read_data(self):
        """
        read the weight data
        :return:
        """
        with open('../data/weight_trained.pkl', 'rb') as f:
            self.trained_weight = pickle.load(f)

    def draw_operator_weight_heat_map(self):
        """
        draw_operator_weight_heat_map
        :return:
        """
        fig, ax = plt.subplots(dpi=200, figsize=(10, 8))
        # normalized_weight = (trained_weight - np.min(trained_weight)) / (np.max(trained_weight)
        # - np.min(trained_weight))
        im = ax.imshow(self.trained_weight, cmap='YlOrRd', aspect='auto')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("交叉权重", fontproperties=font, fontsize=18, rotation=-90, va="bottom")
        cbar.ax.tick_params(labelsize=18)

        ax.set_xticks(np.arange(len(self.repair_operator)), fontproperties=font, fontsize=18, labels=self.repair_operator)
        ax.set_yticks(np.arange(len(self.destroy_operator)), fontproperties=font, fontsize=18, labels=self.destroy_operator)
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="left", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=-60, ha="right", rotation_mode="anchor")
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(self.trained_weight.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(self.trained_weight.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(len(self.destroy_operator)):
            for j in range(len(self.repair_operator)):
                if (i == 1 and j == 1) or (i == 3 and j == 1):
                    text = ax.text(j, i, round(self.trained_weight[i, j], 5), ha="center", va="center", color="white",
                                   fontsize=20)
                else:
                    text = ax.text(j, i, round(self.trained_weight[i, j], 5), ha="center", va="center", color="black",
                                   fontsize=20)

        fig.tight_layout()
        plt.savefig("strategy_weight_heatmap.pdf")
        plt.show()

    def run(self):
        self.read_data()
        self.draw_operator_weight_heat_map()


if __name__ == "__main__":
    drawer = Drawer()
    drawer.run()
