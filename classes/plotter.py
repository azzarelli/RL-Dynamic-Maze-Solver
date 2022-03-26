import matplotlib.pyplot as plt
import numpy as np

class Plotter:

    def __init__(self):
        self.scores = []
        self.scores_avg = []
        self.eps_history = []

    def data_in(self, score):
        self.scores.append(score)
        self.scores_avg.append(np.mean(self.scores[-100:]))

    @property
    def show(self):
        plt.show()

    def live_plot(self):

        plt.figure(1)
        plt.plot(self.scores)
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.savefig('liveplot/score.png')

        plt.figure(2)
        plt.plot(self.scores_avg)
        plt.xlabel('Epochs')
        plt.ylabel('Avg Score')
        plt.savefig('liveplot/avg_score.png')

