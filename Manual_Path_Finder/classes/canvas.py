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
from Manual_Path_Finder.lib.read_maze import load_maze

'''Define size of canvas and location of our reference point'''
SCREENSIZE = W, H = 1400, 1000
mazeWH = 1000
origin = (((W - mazeWH)/2)-150, (H - mazeWH)/2)
lw = 2 # linewidth of drawings made on canvas

'''Colours defined for use in canvas'''
GREY = (140,140,140) # (15,15,15)
DARKGREY = (27, 27,0)
RED = (255, 0, 0)
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

    def step(self, visible, idx, path):
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
        self.draw_visible()

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

    def draw_visible(self):
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


    def get_event(self):
        """Pygame fetch keyboard press events (notably to quit/slow the game down)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    self.slow = 1 if self.slow == 0 else 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                return 1
            elif keys[pygame.K_RIGHT]:
                return 4
            elif keys[pygame.K_DOWN]:
                return 3
            elif keys[pygame.K_LEFT]:
                return 2
            if event.type == pygame.NOEVENT:
                return 0
        return 0