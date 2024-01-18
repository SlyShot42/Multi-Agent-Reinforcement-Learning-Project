import numpy as np
from termcolor import colored
from .config import CELL_NUMBER
from pygame.math import Vector2


class GameObj:
    def __init__(self, *pts: list[Vector2]) -> None:
        self.set_vector(pts)

    def collision(self, other):
        if isinstance(other, GameObj):
            return np.any((self.vector + other.vector) > 1)
        else:
            raise TypeError("unsupported type for collision")

    def set_vector(self, *pts: list[Vector2]):
        self.vector = np.zeros((CELL_NUMBER, CELL_NUMBER))
        for pt in pts:
            self.vector[int(pt.y), int(pt.x)] = 1

    @staticmethod
    def display_vectors(*objects):
        disp = np.zeros((CELL_NUMBER, CELL_NUMBER))
        for obj in objects:
            disp += obj.vector
        for row in disp:
            for col in row:
                if col == 0:
                    print(colored(int(col), "white"), end=" ")
                else:
                    print(colored(int(col), "red"), end=" ")
            print()
        print()
