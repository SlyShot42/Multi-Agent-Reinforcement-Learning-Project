import pygame

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
CELL_SIZE = 40
CELL_NUMBER = 20
clock = pygame.time.Clock()
SCREEN = pygame.display.set_mode((CELL_NUMBER * CELL_SIZE, CELL_NUMBER * CELL_SIZE))
SCREEN_UPDATE = pygame.USEREVENT
