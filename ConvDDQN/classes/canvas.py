"""Visualisation method for game using pygame interface

Notes
-----
Through this method we are looking at displaying the game, the input of Convolutional network and other parameter
values which change from episode, for example values of epsilon or exploration/q-values for choice. If we want to
visually analyse the maze SLOWLY between learning in steps then uncomment `time.sleep(5)` in the learn method in
`agent.py`.
You can also hit 'F1' for trigger a slow down of maze so its easier to visually track changes
"""

import time
import pygame
import sys
import numpy as np
from ConvDDQN.lib.read_maze import load_maze


'''Define size of canvas and location of our reference point'''
SCREENSIZE = W, H = 1400, 1000
mazeWH = 1000
origin = (((W - mazeWH)/2)-150, (H - mazeWH)/2)
lw = 2 # linewidth of drawings made on canvas

'''Colours defined for use in canvas'''
GREY = (140,140,140) # (15,15,15)
DARKGREY = (27, 27,0)
RED = (255, 0, 0)
PINK = (255,120,120)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
DARKGREEN = (0, 150, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 201)

'''Canvas class'''
class Canvas:
    def __init__(self):
        self.step_cntr = 0 # track learn-step counter
        self.cntr = 0 # track loops within canvas display method

        self.maze = load_maze() # load global maze (not seen by actor)
        self.maze = self.maze.T # flip matrix

        self.shape = 201
        self.slow = 0 # choose to slowdown visualisation as they appear

        '''Initialise pygame application and set canvas'''
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 16) # initialise font
        self.surface = pygame.display.set_mode(SCREENSIZE)
        self.actor = (1,1)
        self.optimal_path = np.load('classes/optimal_path.npy')


    def drawSquareCell(self, x, y, dimX, dimY, col=(0, 0, 0)):
        """Method for drawing square blocks for grid-world"""
        pygame.draw.rect(
            self.surface, col,
            (x, y, dimX, dimY)
        )

    def drawSquareGrid(self, origin, gridWH):
        """Method for drawing Global maze grid"""
        CONTAINER_WIDTH_HEIGHT = gridWH # initilised outside class method
        cont_x, cont_y = origin
        # DRAW Grid Border:
        # TOP lEFT TO RIGHT
        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y), lw)
        # # BOTTOM lEFT TO RIGHT
        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, CONTAINER_WIDTH_HEIGHT + cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x,
             CONTAINER_WIDTH_HEIGHT + cont_y), lw)
        # # LEFT TOP TO BOTTOM
        pygame.draw.line(
            self.surface, BLACK,
            (cont_x, cont_y),
            (cont_x, cont_y + CONTAINER_WIDTH_HEIGHT), lw)
        # # RIGHT TOP TO BOTTOM
        pygame.draw.line(
            self.surface, BLACK,
            (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y),
            (CONTAINER_WIDTH_HEIGHT + cont_x,
             CONTAINER_WIDTH_HEIGHT + cont_y), lw)

    def placeCells(self):
        """Draw grid-wrold cells onto canvas"""
        # GET CELL DIMENSIONS...
        cellBorder = 0
        celldimX = celldimY = (mazeWH / self.shape)

        for rows in range(201): # for each cell in the global maze
            for cols in range(201):
                if (self.maze[rows][cols] == 0): # Is the grid cell tiled ?
                    self.drawSquareCell(
                        origin[0] + (celldimY * rows)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimX * cols)
                        + cellBorder + lw / 2,
                        celldimX, celldimY, col=BLACK)
                if cols == 199 and rows == 199:
                    self.drawSquareCell(
                        origin[0] + (celldimY * rows)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimX * cols)
                        + cellBorder + lw / 2,
                        celldimX, celldimY, col=BLUE)
        for s in self.optimal_path: # diplay the path traced by individual
            self.drawSquareCell(
                origin[0] + (celldimX * s[0]) + lw / 2,
                origin[1] + (celldimY * s[1]) + lw / 2,
                celldimX, celldimY, col=PINK)
    def step(self, visible, idx, path, acts, action, score, reward,
             tep_cntr, ep, wall, visit, obs, epsilon, lr,
             gamma=None, rand=None,
             score_split=[], step_split=[],
             alpha=None, beta=None, EP=None, memsize=None, batchsize=None):
        """Run the pygame environment for displaying the maze structure and visible (local) environment of actor
        """
        ''''''
        self.get_event() # fetch keyboard event
        self.set_visible(visible, idx, path) # set new global parameters for tacking path and positions

        '''Draw Background for UI'''
        self.surface.fill(GREY) # canvas background color
        self.drawSquareGrid(origin, mazeWH) # draw the background-grid

        '''Draw Foreground for UI'''
        self.placeCells() # draw global maze
        self.draw_visible(acts, action,  score, reward, tep_cntr, ep,
                          wall, visit, obs, epsilon, lr,
                          gamma=gamma, rand=rand,
                          score_split=score_split, step_split=step_split,
                          alpha=alpha, beta=beta, EP=EP, memsize=memsize, batchsize=batchsize) # handle drawing parameters and local observations

        '''Update pygame and step game'''
        pygame.display.update()
        self.step_cntr += 1

        '''If `F1` is pressed slow game'''
        if self.slow == 1:
            time.sleep(5)
        else:
            pass

    def set_visible(self, visible, idx, path):
        """Set global values for tracking the path and actor position"""
        self.vis = visible
        self.actor = idx
        self.path = path

    def draw_visible(self, acts, action, score, reward, step_cntr, ep, wall,
                     visit, obs, epsilon, lr,
                     gamma=None, rand=None,
                     score_split=[], step_split=[],
                     alpha=None, beta=None,EP=None, memsize=None, batchsize=None):
        """Draw the visible environment around the actor + tracking active parameters

        Notes
        -----
        RED - signifies a fire
        DARKGREY - signifies a visible wall
        WHITE - signifies a path
        GREEN - indicates the actor's position

        Action Space
        ------------
        0 - no move
        1 - up
        2 - left
        3 - down
        4 - right

        """
        '''Visualise the Episode, running score, running total number of steps and current path length of actor'''
        d = 260 # variable which controls the position of parameter tracking on the info menu
        self.surface.blit(self.font.render('Episode: '+str(ep), True, (200, 200, 200)),
                          (1100, d))
        self.surface.blit(self.font.render('Score: ' + str(float('%.5f' % score)) + '  |  ' + str(float('%.5f' % reward)), True,(200, 200, 200)),
                          (1100, d+20))
        self.surface.blit(self.font.render('Total Steps: '+str(step_cntr), True, (200, 200, 200)),
                          (1100, d+40))
        self.surface.blit(self.font.render('Path Len: '+str(len(self.path)), True, (200, 200, 200)),
                          (1100, d+60))

        '''Visualise number of times the actor has stayed/revisited cells/walked into walls'''
        d = d+90 # add distance to paste image of convolutional network image above the following writing
        self.surface.blit(self.font.render('Stay | Visited | Wall', True, (200, 200, 200)),
                          (1100, d))
        if score_split != []: # paste relative scores
            d = d+20
            self.surface.blit(self.font.render(str(float('%.1f' % score_split[0]))+'    '+
                                               str(float('%.1f' % score_split[1]))+'    '+
                                               str(float('%.1f' % score_split[2]))
                                               , True, (200, 200, 200)), (1100, d))
        if step_split != []:
            d = d+20
            self.surface.blit(self.font.render('  '+str(step_split[0])+'       '+
                                               str(step_split[1])+'       '+
                                               str(step_split[2])
                                               , True, (200, 200, 200)), (1100, d))

        '''Visualise gamma, epsilon learning rate, type of experience replay, memory size and batch size'''
        d = d+30
        self.surface.blit(self.font.render('Gamma: ' + str(float('%.5f' % gamma)), True, (200, 200, 200)),
                          (1100, d))
        self.surface.blit(self.font.render('Epsilon: ' + str(float('%.5f' % epsilon)), True, (200, 200, 200)),
                          (1100, d+20))
        self.surface.blit(self.font.render('Learn Rate: ' + str(float('%.5f' % lr)), True, (200, 200, 200)),
                          (1100, d+40))
        self.surface.blit(self.font.render('ExpRep: ' + EP, True, (200, 200, 200)),
                          (1100, d +60))
        self.surface.blit(self.font.render('MemSize: ' + str(memsize), True, (200, 200, 200)),
                          (1100, d +80))
        self.surface.blit(self.font.render('BatchSize: ' + str(batchsize), True, (200, 200, 200)),
                          (1100, d +100))

        '''Visualise the alpha and beta parameters (if they exist, hence if PER is chosens)'''
        d = d+140
        if alpha != None:
            self.surface.blit(self.font.render('Alpha: ' + str(float('%.2f' % alpha)), True, (200, 200, 200)),
                          (1100, d))
        if beta != None:
            self.surface.blit(self.font.render('Beta: ' + str(float('%.4f' % beta)), True, (200, 200, 200)),
                          (1100, d+20))

        '''Visualise the Q-values relative to current action
                Note - to run visualise this you should slow the canvas down, as these values change quickly
        '''
        if acts != []:
            self.surface.blit(self.font.render(str(float('%.3f' % acts[0][2])), True, (200, 200, 200)),
                          (1250, 120))
            self.surface.blit(self.font.render(str(float('%.3f' % acts[0][4])), True, (200, 200, 200)),
                              (1350, 120))
            self.surface.blit(self.font.render(str(float('%.3f' % acts[0][1])), True, (200, 200, 200)),
                              (1300, 80))
            self.surface.blit(self.font.render(str(float('%.3f' % acts[0][3])), True, (200, 200, 200)),
                              (1300, 160))
            self.surface.blit(self.font.render(str(float('%.3f' % acts[0][0])), True, (200, 200, 200)),
                              (1300, 120))

        '''Visualise the Action taken (and if random we still show which action is taken)'''
        # if acts != []:
        # acts = acts.data.cpu().numpy()[0]
        if action == 0:action_str = 'Stay'
        elif action == 1:action_str = 'Up'
        elif action == 2:action_str = 'Left'
        elif action == 3:action_str = 'Down'
        elif action == 4:action_str = 'Right'
        if rand==True: action_str = 'Random ' + action_str
        self.surface.blit(self.font.render('Action: ' + action_str, True, (200, 200, 200)),
                          (1100, 70))

        '''Visualise the input of the convolutional network so we can verify the input is correct'''
        celldimX = celldimY = (mazeWH / self.shape)
        for s in self.path: # diplay the path traced by individual
            self.drawSquareCell(
                origin[0] + (celldimX * s[0]) + lw / 2,
                origin[1] + (celldimY * s[1]) + lw / 2,
                celldimX, celldimY, col=DARKGREEN)

        self.drawSquareCell( # diplsay the actor in the maze
            origin[0] + (celldimX * self.actor[0]) + lw / 2,
            origin[1] + (celldimY * self.actor[1]) + lw / 2,
            celldimX, celldimY, col=GREEN)
        cell_size = 3 # define the cell size shown (can enlarge if too small)
        for row, obs_row in enumerate(obs):
            for col, obs_i in enumerate(obs_row):
                self.drawSquareCell(
                    1100 + (cell_size*(col - 1)),
                    95 + (cell_size*(row - 1)),
                    cell_size, cell_size, col=obs_i)

    def get_event(self):
        """Pygame fetch keyboard press events (notably to quit/slow the game down)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    self.slow = 1 if self.slow == 0 else 0
                if event.key == pygame.K_UP:
                    return 1
                if event.key == pygame.K_RIGHT:
                    return 2
                if event.key == pygame.K_DOWN:
                    return 3
                if event.key == pygame.K_LEFT:
                    return 4