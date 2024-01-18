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
from itertools import combinations
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
        self.generate_entities()

    def generate_entities(self):
        for i in range(self.num_agents):
            # sets the initial positions of the bots
            front = self.positions[-1]
            temp = np.array(self.positions)
            back_array = temp[
                (self.positions == (front.x, front.y - 1))
                | (self.positions == (front.x, front.y + 1))
                | (self.positions == (front.x - 1, front.y))
                | (self.positions == (front.x + 1, front.y))
            ]
            back = back_array[-1]
            self.bots.append(Agent(Vector2(front), Vector2(back)))
            self.move_to_extractions(front, back)

            # sets the initial positions of the end points
            end_pt = self.positions[-1]
            self.end_pts.append(EndPnt(Vector2(*end_pt)))
            self.move_to_extractions(end_pt)
        for bot in self.bots:
            bot.update_state(self)

    def move_to_positions(self, *points):
        for pt in points:
            self.positions.remove(pt)
            self.extractions.append(pt)

    def move_to_extractions(self, *points):
        for pt in points:
            self.extractions.remove(pt)
            self.positions.append(pt)

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
                if bot.collision(self.end_pts[idx]):
                    scores[idx] += 1
                    rewards[idx] = 10
                    # replaces the current position of the end point with a new random position in the positions list
                    self.move_to_positions(
                        (self.end_pts[idx].pos.x, self.end_pts[idx].pos.y)
                    )
                    random.shuffle(self.positions)
                    self.end_pts[idx].place(Vector2(*self.positions[-1]))
                    self.move_to_extractions(self.positions[-1])
                # replaces the old position of the bot with its new position in the positions list
                self.move_to_positions(
                    (prev_front.x, prev_front.y), (prev_back.x, prev_back.y)
                )
                self.move_to_extractions(
                    (bot.front.x, bot.front.y), (bot.back.x, bot.back.y)
                )
            except IndexError:
                bot.move_agent(Action.NO_ACTION)
                dones[idx] = 1
                rewards[idx] = -10
        tracked_bots = []
        for i, j in combinations(range(self.num_agents), 2):
            if self.bots[i].collision(self.bots[j]):
                dones[i] = 1
                dones[j] = 1
                rewards[i] = -10
                rewards[j] = -10
            if i not in tracked_bots:
                tracked_bots.append(i)
                self.bots[i].update_state(self)
            if j not in tracked_bots:
                tracked_bots.append(j)
                self.bots[j].update_state(self)
        return rewards, dones, scores

    def reset(self, dones: list[bool] = None, **kwargs):
        for idx, bot in enumerate(self.bots):
            if kwargs is not None:
                bot.train_short_memory(
                    kwargs["states_old"][idx],
                    kwargs["final_moves"][idx],
                    kwargs["rewards"][idx],
                    kwargs["states_new"][idx],
                    dones[idx],
                )

                # remember
                bot.remember(
                    kwargs["states_old"][idx],
                    kwargs["final_moves"][idx],
                    kwargs["rewards"][idx],
                    kwargs["states_new"][idx],
                    dones[idx],
                )

            if dones[idx]:
                # moves the bot's current position stored in the extraction list back into the positions list
                self.move_to_positions(
                    (bot.front.x, bot.front.y), (bot.back.x, bot.back.y)
                )
                random.shuffle(self.positions)

                # sets a positions of the bots using the positions list
                front = self.positions[-1]
                temp = np.array(self.positions)
                back_array = temp[
                    (self.positions == (front.x, front.y - 1))
                    | (self.positions == (front.x, front.y + 1))
                    | (self.positions == (front.x - 1, front.y))
                    | (self.positions == (front.x + 1, front.y))
                ]
                back = back_array[-1]
                bot.place(front, back)
                self.move_to_extractions(front, back)

                if kwargs is not None:
                    bot.n_games += 1
                    bot.train_long_memory()
                    if kwargs["scores"][idx] > kwargs["records"][idx]:
                        kwargs["records"][idx] = kwargs["scores"][idx]
                    print(
                        f"bot {idx} {bot.n_games}, bot {idx} score: {kwargs['scores'][idx]}"
                    )
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

            self.reset(
                dones=dones,
                states_old=states_old,
                final_moves=final_moves,
                rewards=rewards,
                states_new=states_new,
                scores=scores,
                records=records,
            )
