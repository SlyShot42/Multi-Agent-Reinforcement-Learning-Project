import pygame
import numpy as np
import sys
from pygame.math import Vector2
from .config import CELL_NUMBER
from .config import CELL_SIZE
from .agent import Agent
from .end_pnt import EndPnt
from .agent import Action
from .game_obj import GameObj
import random

pygame.init()
clock = pygame.time.Clock()
SCREEN = pygame.display.set_mode((CELL_NUMBER * CELL_SIZE, CELL_NUMBER * CELL_SIZE))
SCREEN_UPDATE = pygame.USEREVENT
pygame.time.set_timer(SCREEN_UPDATE, 150)


class MainGame:
    def __init__(self, num_agents) -> None:
        self.num_agents = num_agents
        self.positions = [
            (x, y) for x in range(CELL_NUMBER) for y in range(CELL_NUMBER)
        ]
        random.shuffle(self.positions)
        self.bots = []
        self.end_pts = []
        self.extractions = []
        for i in range(self.num_agents):
            # sets the initial positions of the bots
            front = self.positions.pop()
            self.extractions.append(front)
            front = Vector2(*front)
            temp = np.array(self.positions)
            back_array = temp[
                (self.positions == (front.x, front.y - 1))
                | (self.positions == (front.x, front.y + 1))
                | (self.positions == (front.x - 1, front.y))
                | (self.positions == (front.x + 1, front.y))
            ]
            back = self.positions.pop(self.positions.index(back_array[0]))
            self.extractions.append(back)
            back = Vector2(*back_array[0])
            self.bots.append(Agent(front, back))

            # sets the initial positions of the end points
            temp1 = self.positions.pop()
            self.extractions.append(temp1)
            self.end_pts.append(EndPnt(Vector2(*temp1)))
        for bot in self.bots:
            bot.update_state(self)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == SCREEN_UPDATE:
                    state = [bot.state for bot in self.bots]
                    actions = [
                        bot.get_action(state[i]) for i, bot in enumerate(self.bots)
                    ]
                    rewards, dones, scores = self.step(actions)
                    self.reset(dones)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset(np.ones(self.num_agents))

            SCREEN.fill((175, 215, 70))
            self.draw_elements()
            pygame.display.update()
            clock.tick(60)

    """
    check for compatibility with the updated action

    this method will first move the agents according to the input action
    if the bot crashes into a boundary then record a game over for that bot: reward = -10
    if the bot crashes into another bot then record a game over for both bots: reward = -10

    if the bot reaches the end point then update score for that bot: reward = 10
    and place new end point
    
    update the state of each bot
    """

    def step(self, bots_action: list[Action]):
        rewards = np.zeros(self.num_agents)
        dones = np.zeros(self.num_agents)
        scores = np.zeros(self.num_agents)
        for idx, bot in enumerate(self.bots):
            try:
                prev_front = bot.front
                prev_back = bot.back
                bot.move_agent(bots_action[idx])
                collisions = np.zeros(self.num_agents)
                for other_bot in self.bots:
                    if bot is not other_bot:
                        collisions[idx] = bot.collision(other_bot)
                        break
                if bot.collision(self.end_pts[idx]):
                    scores[idx] += 1
                    rewards[idx] = 10
                    # replaces the current position of the end point with a new random position in the positions list
                    self.positions.append(
                        self.extractions[
                            self.extractions.index(
                                (self.end_pts[idx].pos.x, self.end_pts[idx].pos.y)
                            )
                        ]
                    )
                    self.extractions.remove(
                        (self.end_pts[idx].pos.x, self.end_pts[idx].pos.y)
                    )
                    random.shuffle(self.positions)
                    self.end_pts[idx].place(Vector2(*self.positions.pop()))
                elif collisions[idx]:
                    dones[idx] = 1
                    rewards[idx] = -10
                    continue
                # replaces the old position of the bot with its new position in the positions list
                self.positions.append(
                    self.extractions[
                        self.extractions.index((prev_front.x, prev_front.y))
                    ]
                )
                self.positions.append(
                    self.extractions[self.extractions.index((prev_back.x, prev_back.y))]
                )
                self.extractions.remove((prev_front.x, prev_front.y))
                self.extractions.remove((prev_back.x, prev_back.y))
                self.positions.remove((bot.front.x, bot.front.y))
                self.positions.remove((bot.back.x, bot.back.y))
                self.extractions.append((bot.front.x, bot.front.y))
                self.extractions.append((bot.back.x, bot.back.y))
            except IndexError:
                bot.move_agent(Action.NO_ACTION)
                dones[idx] = 1
                rewards[idx] = -10
        for bot in self.bots:
            bot.update_state(self)
        return rewards, dones, scores

    def reset(self, dones: list[bool] = None):
        for idx, bot in enumerate(self.bots):
            if dones[idx]:
                # moves the bot's current position stored in the extraction list back into the positions list
                self.positions.append(
                    self.extractions[self.extractions.index((bot.front.x, bot.front.y))]
                )
                self.positions.append(
                    self.extractions[self.extractions.index((bot.back.x, bot.back.y))]
                )
                self.extractions.remove((bot.front.x, bot.front.y))
                self.extractions.remove((bot.back.x, bot.back.y))
                random.shuffle(self.positions)

                # sets a positions of the bots using the positions list
                front = self.positions.pop()
                self.extractions.append(front)
                front = Vector2(*front)
                temp = np.array(self.positions)
                back_array = temp[
                    (self.positions == (front.x, front.y - 1))
                    | (self.positions == (front.x, front.y + 1))
                    | (self.positions == (front.x - 1, front.y))
                    | (self.positions == (front.x + 1, front.y))
                ]
                back = self.positions.pop(self.positions.index(back_array[0]))
                self.extractions.append(back)
                back = Vector2(*back_array[0])
                bot.place(front, back)
        for bot in self.bots:
            bot.update_state(self)

    def draw_elements(self):
        for idx, bot in enumerate(self.bots):
            bot.draw_agent()
            self.end_pts[idx].draw_end_pnt()

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        records = np.zeros(self.num_agents)
        while True:
            # get old state
            states_old = [bot.state for bot in self.bots]

            # get move
            final_moves = [
                bot.get_action(states_old[i]) for i, bot in enumerate(self.bots)
            ]

            # perform move and get new state
            rewards, dones, scores = self.step(final_moves)
            states_new = [bot.state for bot in self.bots]

            for i, bot in enumerate(self.bots):
                # short term memory
                bot.train_short_memory(
                    states_old[i], final_moves[i], rewards[i], states_new[i], dones[i]
                )

                # remember
                bot.remember(
                    states_old[i], final_moves[i], rewards[i], states_new[i], dones[i]
                )

                if dones[i]:
                    # moves the bot's current position stored in the extraction list back into the positions list
                    self.positions.append(
                        self.extractions[
                            self.extractions.index((bot.front.x, bot.front.y))
                        ]
                    )
                    self.positions.append(
                        self.extractions[
                            self.extractions.index((bot.back.x, bot.back.y))
                        ]
                    )
                    self.extractions.remove((bot.front.x, bot.front.y))
                    self.extractions.remove((bot.back.x, bot.back.y))
                    random.shuffle(self.positions)

                    # sets a positions of the bots using the positions list
                    front = self.positions.pop()
                    self.extractions.append(front)
                    front = Vector2(*front)
                    temp = np.array(self.positions)
                    back_array = temp[
                        (self.positions == (front.x, front.y - 1))
                        | (self.positions == (front.x, front.y + 1))
                        | (self.positions == (front.x - 1, front.y))
                        | (self.positions == (front.x + 1, front.y))
                    ]
                    back = self.positions.pop(self.positions.index(back_array[0]))
                    self.extractions.append(back)
                    back = Vector2(*back_array[0])
                    bot.place(front, back)
                    bot.n_games += 1
                    bot.train_long_memory()
                    if scores[i] > records[i]:
                        records[i] = scores[i]
                    print(f"bot {i} {bot.n_games}, bot {i} score: {scores[i]}")
