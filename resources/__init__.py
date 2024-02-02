from .main_game import MainGame
from .agent import Agent
from .agent import Action
from .end_pnt import EndPnt
from .game_obj import GameObj
from .config import SCREEN_UPDATE

import pygame

pygame.init()
pygame.time.set_timer(SCREEN_UPDATE, 150)
