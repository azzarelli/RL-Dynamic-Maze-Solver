import pygame
import sys

from lib.read_maze import load_maze, get_local_maze_information


SCREENSIZE = W, H = 1400, 1000
mazeWH = 1000
origin = (((W - mazeWH)/2)-150, (H - mazeWH)/2)
lw = 2 # linewidth of maze-grid

# Colors
GREY = (140,140,140) # (15,15,15)
DARKGREY = (27, 27,0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
DARKGREEN = (0, 150, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 201)


class Canvas:
    def __init__(self):
        self.step_cntr = 0
        self.cntr = 0

        self.maze = load_maze()
        self.shape = 201

        pygame.init()
        self.font = pygame.font.SysFont('Arial', 20)
        self.surface = pygame.display.set_mode(SCREENSIZE)
        self.actor = (1,1)

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
        for rows in range(201):
            for cols in range(201):
                # Is the grid cell tiled ?
                if (self.maze[rows][cols] == 0):
                    self.drawSquareCell(
                        origin[0] + (celldimX * cols)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimY * rows)
                        + cellBorder + lw / 2,

                        celldimX, celldimY, col=BLACK)
                if cols == 199 and rows == 199:
                    self.drawSquareCell(
                        origin[0] + (celldimX * cols)
                        + cellBorder + lw / 2,
                        origin[1] + (celldimY * rows)
                        + cellBorder + lw / 2,

                        celldimX, celldimY, col=BLUE)

    def step(self, visible, idx, path, acts, score, steps, walls):
        """Run the pygame environment for displaying the maze structure and visible (local) environment of actor
        """
        self.get_event()
        self.set_visible(visible, idx, path, score)

        self.surface.fill(GREY)
        self.drawSquareGrid(origin, mazeWH)

        self.placeCells()
        self.draw_visible(acts, steps, walls)
        pygame.display.update()
        self.step_cntr += 1

    def set_visible(self, visible, idx, path, score):
        self.vis = visible
        self.actor = idx
        self.path = path
        self.pathlen = len(path)
        self.score = score

    def draw_visible(self, acts, steps, walls):
        """Draw the visible environment around the actor

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
        celldimX = celldimY = (mazeWH / self.shape)

        #self.visible = self.vis
        for s in self.path:
            self.drawSquareCell(
                origin[0] + (celldimX * s[0])
                + lw / 2,
                origin[1] + (celldimY * s[1])
                + lw / 2,

                celldimX, celldimY, col=DARKGREEN)

        self.surface.blit(self.font.render('Score: ' + str(float('%.2f' % self.score)) + '/800', True, (0, 0, 0)),
                          (1100,
                           100))
        self.surface.blit(self.font.render('Len: ' + str(self.pathlen), True, (0, 0, 0)),
                          (1100 ,
                           130))
        self.surface.blit(self.font.render('Steps: ' + str(steps), True, (0, 0, 0)),
                          (1100,
                           160))
        self.surface.blit(self.font.render('Walls: ' + str(walls), True, (0, 0, 0)),
                          (1100,
                           190))

        if acts != []:
            acts = acts.data.cpu().numpy()[0]

        for row in range(3):
            for col in range(3):
                c = self.actor[0] + (col - 1)
                r = self.actor[1] + (row - 1)



                if row == 1 and col == 1:
                    self.drawSquareCell(
                        origin[0] + (celldimX * c)
                        + lw / 2,
                        origin[1] + (celldimY * r)
                        + lw / 2,

                        celldimX, celldimY, col=GREEN)

                else:
                    if self.vis[row][col][1] > 0:
                        pass
                        # self.drawSquareCell(
                        #     origin[0] + (celldimX * c)
                        #     + lw / 2,
                        #     origin[1] + (celldimY * r)
                        #     + lw / 2,
                        #
                        #     celldimX, celldimY, col=RED)
                        #
                        # self.drawSquareCell(
                        #     300 + (40 * (col - 1))
                        #     + 8,
                        #     1200 + (40*(row - 1))
                        #     + 8,
                        #     40, 40, col=RED)
                    elif self.vis[row][col][0] == 0:
                        self.drawSquareCell(
                            1200 + (40 * (row - 1))
                            + 8,
                            300 + (40 * (col - 1))
                            + 8,
                            40, 40, col=DARKGREY)


    def get_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()