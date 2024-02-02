from pygame.math import Vector2
from .game_obj import GameObj
from .config import CELL_SIZE
from .config import SCREEN
import pygame


class EndPnt(GameObj):
    def __init__(self, pos: Vector2) -> None:
        self.place(pos)

    def draw_end_pnt(self):
        terminal_rect = pygame.Rect(
            int(self.pos.x * CELL_SIZE),
            int(self.pos.y * CELL_SIZE),
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(SCREEN, (255, 0, 162), terminal_rect)

    def place(self, pos: Vector2):
        self.pos = pos
        super().set_vector(self.pos)

    def __eq__(self, other):
        if isinstance(other, EndPnt):
            return self.pos == other.pos
        else:
            raise TypeError("unsupported type for equality")
