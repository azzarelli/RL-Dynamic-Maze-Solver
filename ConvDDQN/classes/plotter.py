"""Graphing Handler for tracking loss, path length, etc.
"""

import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class Plotter:
    def __init__(self, name):
        '''Initialise the save-name and trackers'''
        self.name = name

        self.scores = []
        self.loss = []
        self.scores_avg = []
        self.eps_history = []

        self.wall_cntrs = []
        self.stay_cntrs = []
        self.visit_cntrs = []
        self.path_cntrs = []


    def data_in(self, score, loss, wall_cntr=-1, stay_cntr=-1, visit_cntr=-1, path_cntr=-1, epsilon=-1):
        """Fetch data to save - some values are optional to track"""
        self.loss = self.loss + loss
        self.scores = np.concatenate((self.scores,score))
        self.scores_avg.append(self.scores)

        if wall_cntr > -1:
            self.wall_cntrs.append(wall_cntr)
        if stay_cntr > -1:
            self.stay_cntrs.append(stay_cntr)
        if visit_cntr > -1:
            self.visit_cntrs.append(visit_cntr)
        if path_cntr > -1:
            self.path_cntrs.append(path_cntr)
        self.eps_history.append(epsilon)

        '''Option to dump saved data into folder (for documentation)'''
        # with open('testing_helpers/'+self.name+'.json', 'w') as jf:
            # json.dump(self.scores, jf)

    @property
    def show(self):

        plt.show()

    def plot_socre(self):
        plt.figure()
        plt.plot(self.scores, color='green', label='Raw', alpha=0.5)
        plt.xlabel('Epochs')
        plt.legend()
        plt.ylabel('Scoes')
        plt.savefig('liveplot/Score_' + self.name + '.png')
        plt.clf()
        plt.close('all')

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss, color='green', label='Raw', alpha=0.5)
        #plt.plot(self.scores_avg, color='orange', label='Average', alpha=0.5)
        plt.xlabel('Epochs')
        plt.legend()
        plt.ylabel('Lss')
        plt.savefig('liveplot/Loss_' + self.name + '.png')
        plt.clf()
        plt.close('all')

    def plot_epsilon(self):
        plt.figure()
        plt.plot(self.eps_history, color='green', label='epsilon')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('#')
        plt.savefig('liveplot/Epsilon_' + self.name + '.png')
        plt.clf()
        plt.close('all')

    def plt_pathlength(self):
        plt.figure()
        plt.plot(self.path_cntrs, color='green', label='epsilon')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('#')
        plt.savefig('liveplot/Path_' + self.name + '.png')
        plt.clf()
        plt.close('all')

    def live_plot(self):
        """Plot the data live into liveplot folder"""
        mpl.use("agg")
        print('Plotting ...')

        '''Choose which plotting functionsto call'''
        self.plot_socre()
        self.plot_loss()
        # self.plot_epsilon()
        self.plt_pathlength()

        plt.clf() # need to close plots to prevent memory failures from matplotlib library
        plt.close('all')