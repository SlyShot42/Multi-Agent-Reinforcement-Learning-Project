import numpy as np
import pygame
from pygame.math import Vector2
import torch
import random
from enum import Enum
from collections import deque
from .config import CELL_SIZE
from .config import SCREEN
from .config import MAX_MEMORY
from .config import BATCH_SIZE
from .game_obj import GameObj
from icecream import ic

# from .main_game impor


class Action(Enum):
    # RIGHT =  Vector2(1,0)
    # LEFT = Vector2(-1,0)
    # UP = Vector2(0,-1)
    # DOWN = Vector2(0,1)
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    NO_ACTION = 3


class Agent(GameObj):
    def __init__(self, front: Vector2, back: Vector2) -> None:
        self.place(front, back)
        self.score = 0
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None  # TODO
        self.trainer = None  # TODO

    def draw_agent(self):
        for idx, block in enumerate(self.body):
            x_pos = int(block.x * CELL_SIZE)
            y_pos = int(block.y * CELL_SIZE)
            block_rect = pygame.Rect(x_pos, y_pos, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(
                SCREEN,
                (idx == 0) * (56, 245, 237) + (idx == 1) * (126, 166, 114),
                block_rect,
            )

    def one_hot_encoder(self, action: Action):
        vect = np.zeros(len(Action))
        vect[action.value] = 1
        return vect

    def one_hot_decoder(self, vect: np.array):
        return Action(np.argmax(vect))

    def move_agent(self, action: Action):
        next_loc = 0
        if action == Action.NO_ACTION:
            return
        elif action == Action.FORWARD:
            next_loc = self.direction
        elif action == Action.LEFT:
            next_loc = self.direction.rotate(-90)
        elif action == Action.RIGHT:
            next_loc = self.direction.rotate(90)
        self.place(self.direction + self.back, next_loc + self.front)

    def __eq__(self, other):
        if isinstance(other, Agent):
            return self.front == other.front and self.back == other.back
        else:
            raise TypeError("unsupported type for equality")

    def place(self, front: Vector2, back: Vector2):
        super().set_vector(front, back)
        self.front = front
        self.back = back
        self.body = [self.front, self.back]
        self.direction = self.front - self.back

    def get_action(self, state) -> list[Action]:
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_state(self, game):
        self.state = []
        self.state.append(0)  # danger straight 0
        self.state.append(0)  # danger left 1
        self.state.append(0)  # danger right 2
        for bot in game.bots:
            if sum(self.state) == 3:
                break

            # danger straight
            try:
                straight = self.front + self.direction
                straight = GameObj(straight)
            except IndexError:
                self.state[0] = 1
                continue

            # danger left
            try:
                left = self.front + self.direction.rotate(-90)
                left = GameObj(left)
            except IndexError:
                self.state[1] = 1
                continue

            # danger right
            try:
                right = self.front + self.direction.rotate(90)
                right = GameObj(right)
            except IndexError:
                self.state[2] = 1
                continue
                
            if bot is self:
                continue
            self.state[0] = straight.collision(bot)
            self.state[1] = left.collision(bot)
            self.state[2] = right.collision(bot)

        directions = np.array(
            [
                Vector2(-1, 0),  # left 3
                Vector2(1, 0),  # right 4
                Vector2(0, -1),  # up 5
                Vector2(0, 1),  # down 6
            ]
        )

        # Update direction state
        self.state.extend([int(self.direction == direction) for direction in directions])

        end_pt = game.end_pts[int(np.argmax(np.array(game.bots) == self))]
        if self.direction == directions[0]:  # <-- left
            self.state.append(int(self.front.y < end_pt.pos.y))  # end point left 7
            self.state.append(int(self.front.y > end_pt.pos.y))  # end point right 8
            self.state.append(int(self.front.x > end_pt.pos.x))  # end point straight 9
            self.state.append(int(self.front.x < end_pt.pos.x))  # end point behind 10
        elif self.direction == directions[1]:  # --> right
            self.state.append(int(self.front.y > end_pt.pos.y))  # end point left
            self.state.append(int(self.front.y < end_pt.pos.y))  # end point right
            self.state.append(int(self.front.x < end_pt.pos.x))  # end point straight
            self.state.append(int(self.front.x > end_pt.pos.x))  # end point behind
        elif self.direction == directions[2]:  # ^ up
            self.state.append(int(self.front.x > end_pt.pos.x))  # end point left
            self.state.append(int(self.front.x < end_pt.pos.x))  # end point right
            self.state.append(int(self.front.y > end_pt.pos.y))  # end point straight
            self.state.append(int(self.front.y < end_pt.pos.y))  # end point behind
        elif self.direction == directions[3]:  # v down
            self.state.append(int(self.front.x < end_pt.pos.x))  # end point left
            self.state.append(int(self.front.x > end_pt.pos.x))  # end point right
            self.state.append(int(self.front.y < end_pt.pos.y))  # end point straight
            self.state.append(int(self.front.y > end_pt.pos.y))  # end point behind
