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
        self.positions = [
            (x, y) for x in range(CELL_NUMBER) for y in range(CELL_NUMBER)
        ]
        random.shuffle(self.positions)
        self.bots = []
        self.extractions = []
        for i in range(self.num_agents):
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
        self.end_pts = []
        for i in range(self.num_agents):
            temp = self.positions.pop()
            self.extractions.append(temp)
            self.end_pts.append(EndPnt(Vector2(*temp)))
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
        rewards = np.zeros(self.num_agents)
        dones = np.zeros(self.num_agents)
        scores = np.zeros(self.num_agents)
        for idx, bot in enumerate(self.bots):
            try:
                bot.move_agent(bots_action[idx])
                collisions = np.zeros(self.num_agents)
                for other_bot in self.bots:
                    if bot is not other_bot:
                        collision[idx] = bot.collision(other_bot)
                        break
                if bot.collision(self.end_pts[idx]):
                    scores[idx] += 1
                    rewards[idx] = 10
                    # TODO: replace the old position of the end point with its new position in the positions list
                elif collisions[idx]:
                    dones[idx] = 1
                    rewards[idx] = -10
                    continue
                # TODO: replace the old position of the bot with its new position in the positions list
            except IndexError as e:
                bot.move_agent(Action.NO_ACTION)
                dones[idx] = 1
                rewards[idx] = -10
        for bot in self.bots:
            bot.update_state(self)
        return rewards, dones, scores

    """
    TODO: this method needs to check for dones and reset the corresponding bot's front and back positions using the
    popped positions from the positions list after has been randomly shuffled. 
    """

    def reset(self, dones: list[bool] = None):
        if dones is None:
            dones = np.zeros(self.num_agents)
        for idx, bot in enumerate(self.bots):
            if dones[idx] == 1:
                bot.reset(
                    Vector2(*self.positions.pop()), Vector2(*self.positions.pop())
                )
                dones[idx] = 0
        for idx, end_pt in enumerate(self.end_pts):
            if dones[idx] == 1:
                end_pt.pos = Vector2(*self.positions.pop())
        self.update_state()

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
