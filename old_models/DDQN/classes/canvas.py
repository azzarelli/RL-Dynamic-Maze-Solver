import pygame
import sys

from old_models.DDQN.lib.read_maze import load_maze

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
        self.maze = self.maze.T
        #print(type(self.maze))
        self.shape = 201

        pygame.init()
        self.font = pygame.font.SysFont('Arial', 16)

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

    def step(self, visible, idx, path, acts, action, score, reward, tep_cntr, ep, wall, visit, obs):
        """Run the pygame environment for displaying the maze structure and visible (local) environment of actor
        """
        self.get_event()
        self.set_visible(visible, idx, path)

        self.surface.fill(GREY)
        self.drawSquareGrid(origin, mazeWH)

        self.placeCells()
        self.draw_visible(acts, action,  score, reward, tep_cntr, ep, wall, visit, obs)
        pygame.display.update()
        self.step_cntr += 1

    def set_visible(self, visible, idx, path):
        self.vis = visible
        self.actor = idx
        self.path = path

    def draw_visible(self, acts, action, score, reward, step_cntr, ep, wall, visit, obs):
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
        self.surface.blit(self.font.render('Episode: '+str(ep), True, (200, 200, 200)),
                          (1100, 100))
        self.surface.blit(self.font.render('Steps: '+str(step_cntr), True, (200, 200, 200)),
                          (1100, 130))
        self.surface.blit(self.font.render('Path Len: '+str(len(self.path)), True, (200, 200, 200)),
                          (1100, 160))

        self.surface.blit(self.font.render('Score: ' + str(float('%.2f' % score))+' + '+ str(float('%.2f' % reward)), True, (200, 200, 200)),
                          (1100, 190))
        self.surface.blit(self.font.render('Walls: ' + str(wall), True,(200, 200, 200)),
            (1100, 220))
        self.surface.blit(self.font.render('Revisited: ' + str(visit), True,(200, 200, 200)),
            (1100, 250))

        if acts != []:
            # acts = acts.data.cpu().numpy()[0]
            if action == 0:action_str = 'Stay'
            elif action == 1:action_str = 'Up'
            elif action == 2:action_str = 'Left'
            elif action == 3:action_str = 'Down'
            elif action == 4:action_str = 'Right'
        else: action_str = 'Random'
        self.surface.blit(self.font.render('Action: ' + action_str, True, (200, 200, 200)),
                          (1100, 70))

        celldimX = celldimY = (mazeWH / self.shape)

        #self.visible = self.vis
        for s in self.path:
            self.drawSquareCell(
                origin[0] + (celldimX * s[0]) + lw / 2,
                origin[1] + (celldimY * s[1]) + lw / 2,
                celldimX, celldimY, col=DARKGREEN)

        self.drawSquareCell(
            origin[0] + (celldimX * self.actor[0]) + lw / 2,
            origin[1] + (celldimY * self.actor[1]) + lw / 2,
            celldimX, celldimY, col=GREEN)

        cell_size = 12

        for row, obs_row in enumerate(obs):
            for col, obs_i in enumerate(obs_row):
                c = self.actor[0] + (col - 1)
                r = self.actor[1] + (row - 1)

                self.drawSquareCell(
                    1100 + (cell_size*(col - 1)),
                    350 + (cell_size*(row - 1)),
                    cell_size, cell_size, col=obs_i)

                # if row == 1 and col == 1:
                #     self.drawSquareCell(
                #         1200+ 8,
                #         300+ 8,
                #         40, 40, col=GREEN)
                # else:
                #     if self.vis[row][col][1] > 0:
                #         pass
                #         # self.drawSquareCell(
                #         #     origin[0] + (celldimX * c) + lw / 2,
                #         #     origin[1] + (celldimY * r)+ lw / 2,
                #         #     celldimX, celldimY, col=RED)
                #         #
                #         # self.drawSquareCell(
                #         #     1200 + (40 * (col - 1)) + 8,
                #         #     300 + (40*(row - 1))+ 8,
                #         #     40, 40, col=RED)
                #     elif self.vis[row][col][0] == 0:
                #         self.drawSquareCell(
                #             1200 + (40 * (col - 1)) + 8,
                #             300 + (40 * (row - 1))+ 8,
                #             40, 40, col=DARKGREY)
                #     elif (c, r) in self.path:
                #         self.drawSquareCell(
                #             1200 + (40 * (col - 1)) + 8,
                #             300 + (40 * (row - 1))+ 8,
                #             40, 40, col=DARKGREEN)

    def get_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()