import json

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class Plotter:

    def __init__(self, name):
        self.name = name

        self.scores = []
        self.scores_avg = []
        self.eps_history = []

        self.wall_cntrs = []
        self.stay_cntrs = []
        self.visit_cntrs = []
        self.path_cntrs = []

    def data_in(self, score, wall_cntr=-1, stay_cntr=-1, visit_cntr=-1, path_cntr=-1):
        self.scores.append(score)
        self.scores_avg.append(np.mean(self.scores[-100:]))
        if wall_cntr > -1:
            self.wall_cntrs.append(wall_cntr)
        if stay_cntr > -1:
            self.stay_cntrs.append(stay_cntr)
        if visit_cntr > -1:
            self.visit_cntrs.append(visit_cntr)
        if path_cntr > -1:
            self.path_cntrs.append(path_cntr)

        with open('testing_helpers/'+self.name+'.json', 'w') as jf:
            json.dump(self.scores, jf)

    @property
    def show(self):
        plt.show()

    def live_plot(self):
        mpl.use("agg")

        print('Plotting ...')
        plt.figure()
        plt.plot(self.scores, color='blue', label='Raw')
        plt.plot(self.scores_avg, color='orange', label='Average')
        plt.xlabel('Epochs')
        plt.legend()
        plt.ylabel('Avg Score')
        plt.savefig('liveplot/Score_'+self.name+'.png')

        if self.stay_cntrs != [] and self.visit_cntrs != [] and self.wall_cntrs != []:
            plt.figure()
            plt.plot(self.visit_cntrs, color='green', label='# visted')
            plt.plot(self.stay_cntrs, color='orange', label='# Stay')
            plt.plot(self.wall_cntrs, color='lightblue', label='# Walls')
            plt.plot(self.path_cntrs, color='red', label='Path Length')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('#')
            plt.savefig('liveplot/MonitorCounters_'+self.name+'.png')

        plt.clf()
        plt.close('all')