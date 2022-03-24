import sys
import pygame
from pygame.locals import KEYDOWN, K_q
import numpy as np

from lib.read_maze import load_maze

# CONSTANTS:
SCREENSIZE = WIDTH, HEIGHT = 1100, 1100
BLACK = (0, 0, 0)
RED = (255, 0 ,0 )
GREY = (250, 250, 250)
# OUR GRID MAP:
maze = load_maze()

cellMAP = maze # np.random.randint(2, size=(199, 199))

_VARS = {'surf': False, 'gridWH':1000,
         'gridOrigin': (50, 50), 'gridCells': cellMAP.shape[0], 'lineWidth': 2}

def init_canv():
    pygame.init()
    _VARS['surf'] = pygame.display.set_mode(SCREENSIZE)
    while True:
        checkEvents()
        _VARS['surf'].fill(GREY)
        drawSquareGrid(
        _VARS['gridOrigin'], _VARS['gridWH'], _VARS['gridCells'])
        placeCells()
        pygame.display.update()


# NEW METHOD FOR ADDING CELLS :
def placeCells():
    # GET CELL DIMENSIONS...
    cellBorder = 0
    celldimX = celldimY = (_VARS['gridWH']/_VARS['gridCells']) - (cellBorder*2)
    # DOUBLE LOOP
    for row in range(cellMAP.shape[0]):
        for column in range(cellMAP.shape[1]):
            # Is the grid cell tiled ?
            if(cellMAP[column][row] == 0):
                drawSquareCell(
                    _VARS['gridOrigin'][0] + (celldimY*row)
                    + cellBorder + (2*row*cellBorder) + _VARS['lineWidth']/2,
                    _VARS['gridOrigin'][1] + (celldimX*column)
                    + cellBorder + (2*column*cellBorder) + _VARS['lineWidth']/2,
                    celldimX, celldimY)
            if column == 199 and row == 199:
                drawSquareCell(
                    _VARS['gridOrigin'][0] + (celldimY * row)
                    + cellBorder + (2 * row * cellBorder) + _VARS['lineWidth'] / 2,
                    _VARS['gridOrigin'][1] + (celldimX * column)
                    + cellBorder + (2 * column * cellBorder) + _VARS['lineWidth'] / 2,
                    celldimX, celldimY, col=(255, 0, 0))
            if column == 1 and row == 1:
                drawSquareCell(
                    _VARS['gridOrigin'][0] + (celldimY * row)
                    + cellBorder + (2 * row * cellBorder) + _VARS['lineWidth'] / 2,
                    _VARS['gridOrigin'][1] + (celldimX * column)
                    + cellBorder + (2 * column * cellBorder) + _VARS['lineWidth'] / 2,
                    celldimX, celldimY, col=(0, 255, 0))

# Draw filled rectangle at coordinates
def drawSquareCell(x, y, dimX, dimY, col=(0,0,0)):
    pygame.draw.rect(
     _VARS['surf'], col,
     (x, y, dimX, dimY)
    )


def drawSquareGrid(origin, gridWH, cells):

    CONTAINER_WIDTH_HEIGHT = gridWH
    cont_x, cont_y = origin

    # DRAW Grid Border:
    # TOP lEFT TO RIGHT
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (cont_x, cont_y),
      (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y), _VARS['lineWidth'])
    # # BOTTOM lEFT TO RIGHT
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (cont_x, CONTAINER_WIDTH_HEIGHT + cont_y),
      (CONTAINER_WIDTH_HEIGHT + cont_x,
       CONTAINER_WIDTH_HEIGHT + cont_y), _VARS['lineWidth'])
    # # LEFT TOP TO BOTTOM
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (cont_x, cont_y),
      (cont_x, cont_y + CONTAINER_WIDTH_HEIGHT), _VARS['lineWidth'])
    # # RIGHT TOP TO BOTTOM
    pygame.draw.line(
      _VARS['surf'], BLACK,
      (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y),
      (CONTAINER_WIDTH_HEIGHT + cont_x,
       CONTAINER_WIDTH_HEIGHT + cont_y), _VARS['lineWidth'])


def checkEvents():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == KEYDOWN and event.key == K_q:
            pygame.quit()
            sys.exit()


if __name__ == '__main__':
    main()