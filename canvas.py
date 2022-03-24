import pygame
import sys
import time

from lib.read_maze import load_maze, get_local_maze_information

SCREENSIZE = W, H = 1000, 1000
mazeWH = 1000
origin = ((W - mazeWH)/2, (H - mazeWH)/2)
lw = 2 # linewidth of maze-grid

# Colors
GREY = (15,15,15)
DARKGREY = (27, 27,0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 201)

class Canvas:
    def __init__(self):
        self.step_cntr = 0
        self.maze = load_maze()
        self.shape = self.maze.shape[0]

        pygame.init()
        self.surface = pygame.display.set_mode(SCREENSIZE)

        self.actor = (1,1)
        self.set_visible(get_local_maze_information(*self.actor), (self.actor))

    def drawSquareCell(self, x, y, dimX, dimY, col=(0, 0, 0)):
        pygame.draw.rect(
            self.surface, col,
            (x, y, dimX, dimY)
        )

    def drawSquareGrid(self, origin, gridWH):
        CONTAINER_WIDTH_HEIGHT = gridWH
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
        # GET CELL DIMENSIONS...
        cellBorder = 0
        celldimX = celldimY = (mazeWH / self.shape)

        # DOUBLE LOOP
        for row in range(self.maze.shape[0]):
            for column in range(self.maze.shape[1]):
                # Is the grid cell tiled ?
                if (self.maze[row][column] == 0):
                    self.drawSquareCell(
                        origin[0] + (celldimY * row)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimX * column)
                        + cellBorder + lw / 2,
                        celldimX, celldimY)
                if column == 199 and row == 199:
                    self.drawSquareCell(
                        origin[0] + (celldimY * row)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimX * column)
                        + cellBorder + lw / 2,
                        celldimX, celldimY, col=BLUE)

    def step_canvas(self):
        """Run the pygame environment for displaying the maze structure and visible (local) environment of actor
        """
        self.get_event()

        self.surface.fill(GREY)
        self.drawSquareGrid(origin, mazeWH)

        self.placeCells()
        self.draw_visible()

        pygame.display.update()
        self.step_cntr += 1

    def set_visible(self, visible, idx):
        self.vis = visible
        self.visidx = idx

    def draw_visible(self):
        """Draw the visible environment around the actor

        Notes
        -----
        RED - signifies a fire
        DARKGREY - signifies a visible wall
        WHITE - signifies a path
        GREEN - indicates the actor's position
        """
        celldimX = celldimY = (mazeWH / self.shape)
        self.visible = self.vis
        for row in range(3):
            for col in range(3):
                r = self.visidx[0] + (row - 1)
                c = self.visidx[1] + (col - 1)

                if row == 1 and col == 1:
                    self.drawSquareCell(
                        origin[0] + (celldimY * r)
                        + lw / 2,
                        origin[1] + (celldimX * c)
                        + lw / 2,
                        celldimX, celldimY, col=GREEN)
                else:
                    if self.vis[row][col][1] > 0:

                        self.drawSquareCell(
                            origin[0] + (celldimY * r)
                            + lw / 2,
                            origin[1] + (celldimX * c)
                            + lw / 2,
                            celldimX, celldimY, col=RED)
                    else:
                        if self.vis[row][col][0] == 1:
                            self.drawSquareCell(
                                origin[0] + (celldimY * r)
                                + lw / 2,
                                origin[1] + (celldimX * c)
                                + lw / 2,
                                celldimX, celldimY, col=WHITE)
                        elif self.vis[row][col][0] == 0:
                            self.drawSquareCell(
                                origin[0] + (celldimY * r)
                                + lw / 2,
                                origin[1] + (celldimX * c)
                                + lw / 2,
                                celldimX, celldimY, col=DARKGREY)

    @property
    def get_actor_pos(self):
        return self.actor

    def get_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()