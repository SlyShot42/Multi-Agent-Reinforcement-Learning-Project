import pygame
import numpy as np
from pygame.math import Vector2
from itertools import combinations
from .config import CELL_NUMBER
from .config import SCREEN
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
pygame.time.set_timer(self.SCREEN_UPDATE, 150)


class MainGame:
    def __init__(self, num_agents) -> None:
        self.num_agents = num_agents
        self.reset()
        self.collision = np.zeros((self.num_agents, self.num_agents))
        self.dones = np.zeros(self.num_agents)
        self.positions = [
            (x, y) for x in range(CELL_NUMBER) for y in range(CELL_NUMBER)
        ]
        random.shuffle(self.positions)
        self.bots = []
        for i in range(self.num_agents):
            front = Vector2(*self.positions.pop())
            positions1 = np.array(self.positions)
            back_array = positions1[
                (self.positions == (front.x, front.y - 1))
                | (self.positions == (front.x, front.y + 1))
                | (self.positions == (front.x - 1, front.y))
                | (self.positions == (front.x + 1, front.y))
            ]
            self.positions.remove(back_array[0])
            back = Vector2(*back_array[0])
            self.bots.append(Agent(front, back))
        self.end_pts = []
        for i in range(self.num_agents):
            self.end_pts.append(EndPnt(Vector2(*self.positions.pop())))
        for bot in self.bots:
            bot.update_state(self)

    def run(self):
        while True:
            # action = [Action.RIGHT, Action.LEFT]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == SCREEN_UPDATE:
                    # TODO: in this elif statement,
                    # TODO: the action is to be calculated by the agent
                    actions = ...
                    # TODO: and then passed into the step method
                    self.step(actions)
                    # TODO: check for dones and reset accordingly

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        print("reset")
                        main_game.reset(main_game.num_agents)

            SCREEN.fill((175, 215, 70))
            self.draw_elements()
            pygame.display.update()
            clock.tick(60)

    """
    TODO: return calculated rewards, dones, scores
    check for compatibility with the updated action

    this method will first move the agents according to the input action
    if the bot crashes into a boundary then record a game over for that bot: reward = -10
    if the bot crashes into another bot then record a game over for both bots: reward = -10

    if the bot reaches the end point then update score for that bot: reward = 10
    and place new end point
    
    update the state of each bot
    """

    def step(self, bots_action: list[Action]):
        for idx, bot in enumerate(self.bots):
            try:
                bot.move_agent(bots_action[idx])
                if bot.collision(self.end_pts[i]):
                    self.dones[idx] = 1
            except IndexError as e:
                bot.move_agent(Action.NO_ACTION)
                self.dones[idx] = 1
        if boundaries == len(self.bots):
            print("boundary")
            pygame.time.delay(1000)
            main_game.__init__(self.num_agents)
        else:
            self.update_state()
        # return rewards, dones, scores

    def draw_elements(self):
        for idx, bot in enumerate(self.bots):
            bot.draw_agent()
            self.end_pts[idx].draw_end_pnt()

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = np.zeros(num_agents)
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

            # short term memory
            for i, bot in enumerate(self.bots):
                bot.train_short_memory(
                    states_old[i], final_moves[i], rewards[i], states_new[i], dones[i]
                )

            # remember
            for i, bot in enumerate(self.bots):
                bot.remember(
                    states_old[i], final_moves[i], rewards[i], states_new[i], dones[i]
                )

            # TODO: this needs to check which bots are done and run the corresponding training logic
            if np.sum(dones) == num_agents:
                self.reset()
                for bot in self.bots:
                    bot.n_games += 1
                for bot in self.bots:
                    bot.train_long_memory()

                for i, score in scores:
                    if score > record[i]:
                        record[i] = score

                print(f"Game: {self.bots[0].n_games}, Scores: {scores}")
