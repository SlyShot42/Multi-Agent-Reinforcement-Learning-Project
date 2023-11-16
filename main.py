import pygame, sys
from pygame.math import Vector2
from random import randint
import random
from enum import Enum
import numpy as np
from collections import deque
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from termcolor import colored

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
cell_size = 40
cell_number = 20

'''
TODO: may want to change action to simply forward, left, right, nothing
'''
class Action(Enum):
    # RIGHT =  Vector2(1,0)
    # LEFT = Vector2(-1,0)
    # UP = Vector2(0,-1)
    # DOWN = Vector2(0,1)
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    NO_ACTION = 3

'''
this class is the parent class to all game entities
methods:
    collision(self, other): checks if the object is colliding with another input object
    set_vector(self, pts: list[Vector2]): sets the object vector attribute
'''
class GAME_OBJ:
    def __init__(self,pts: list[Vector2] = []) -> None:
        self.set_vector(pts)
    
    def collision(self, other):
        if isinstance(other,GAME_OBJ):
            return np.any((self.vector + other.vector) > 1)
        else:
            raise TypeError('unsupported type for collision')
        
    def set_vector(self,pts: list[Vector2] = []):
        self.vector = np.zeros((cell_number,cell_number))
        for pt in pts:
            self.vector[int(pt.y),int(pt.x)] = 1

    @staticmethod
    def display_vectors(objects: []):
        disp = np.zeros((cell_number,cell_number))
        for obj in objects:
            disp += obj.vector
        for row in disp:
            for col in row:
                if col == 0:
                    print(colored(int(col),'white'),end=' ')
                else:
                    print(colored(int(col),'red'),end=' ')
            print()
        print()


class Agent(GAME_OBJ):
    '''
    TODO: Q: how exactly will these variables be updated? A:
    '''
    def __init__(self,front: Vector2,back: Vector2) -> None:
        self.front = front
        self.back = back
        self.body = [self.front,self.back]
        super().__init__(self.body)
        self.direction = self.front - self.back
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None # TODO
        self.trainer = None # TODO

    def draw_agent(self):
        for idx, block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect  = pygame.Rect(x_pos,y_pos,cell_size,cell_size)
            pygame.draw.rect(screen,(idx == 0) * (56,245,237) + (idx == 1) * (126,166,114),block_rect)

    def one_hot_encoder(self, action: Action):
        vect = np.zeros(len(Action))
        vect[action.value]
        return vect
    
    def one_hot_decoder(self, vect: np.array):
        return Action(np.argmax(vect))
    
    def move_agent(self,action: Action):
        next_loc = 0
        if action == Action.NO_ACTION:
            return
        elif action == Action.FORWARD:
            next_loc = self.direction
        elif action == Action.LEFT:
            next_loc = self.direction.rotate(-90)
        elif action == Action.RIGHT:
            next_loc = self.direction.rotate(90)
        self.back = self.direction + self.back
        self.front += next_loc
        self.body = [self.front,self.back]
        super().set_vector(self.body)
        self.direction = self.front - self.back

    def __eq__(self, other):
        if isinstance(other,Agent):
            return self.front == other.front and self.back == other.back
        else:
            raise TypeError('unsupported type for equality')

    '''
    FIXME: check and fix the following methods to comply with the single agent model structure
    '''
    def get_action(self, state) -> list[Action]:
        self.epsilon = 80 - self.n_games

        # TODO: the final_move variable must be formatted such that it reflects the Action enum
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = randint(0,2)
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
        self.memory.append(state, action, reward, next_state, done)

    '''
    FIXME: fix this method so that it uses the inherited collide method to check for 
    collisions between game object items. utilize the information in the main game 
    object to update state variable each agent
    '''
    def update_state(self,game):
        self.state = []
        '''
        TODO: this could section should loop through all game objects 
        and check for possible collisions straight, left, and right based
        of the current direction of the agent. 
        '''
        self.state.append(0) # danger straight
        self.state.append(0) # danger left
        self.state.append(0) # danger right
        for idx, bot in enumerate(game.bots):
            if self == bot:
                continue
            if sum(self.state) == 3:
                break

            # danger straight
            try:
                straight = GAME_OBJ([bot.front+bot.direction])
                if self.state[0] == 0:
                    self.state[0] = straight.collision(bot)
            except IndexError as e:
                self.state[0] = 1

            # danger left
            try:
                right = GAME_OBJ([bot.front+bot.direction.rotate(-90)])
                if self.state[1] == 0:
                    self.state[1] = right.collision(bot)
            except IndexError as e:
                self.state[1] = 1

            # danger right
            try:
                left = GAME_OBJ([bot.front+bot.direction.rotate(90)])
                if self.state[2] == 0:
                    self.state[2] = left.collision(bot)
            except IndexError as e:
                self.state[2] = 1

        self.state.append(self.direction == Vector2(-1,0)) # direction left
        self.state.append(self.direction == Vector2(1,0)) # direction right
        self.state.append(self.direction == Vector2(0,-1)) # direction up
        self.state.append(self.direction == Vector2(0,1)) # direction down

        end_pt = game.end_pts[int(np.argmax(np.array(game.bots) == self))]
        '''
        TODO: we want this to check whether end_pt is straight, left, right,
        or behind based off the orientation and where the bot is currently
        '''
        self.state.append(bot.front.x > end_pt.pos.x) # end point left 
        self.state.append(bot.front.x < end_pt.pos.x) # end point right
        self.state.append(bot.front.y > end_pt.pos.y) # end point up
        self.state.append(bot.front.y < end_pt.pos.y) # end point down

class END_PNT(GAME_OBJ):
    def __init__(self,pos: Vector2) -> None:
        self.pos = pos
        super().__init__([self.pos])

    def draw_end_pnt(self):
        terminal_rect = pygame.Rect(int(self.pos.x*cell_size),int(self.pos.y*cell_size),cell_size,cell_size)
        pygame.draw.rect(screen,(255,0,162),terminal_rect)

    def __eq__(self, other):
        if isinstance(other,END_PNT):
            return self.pos == other.pos
        else:
            raise TypeError('unsupported type for equality')
        
class MAIN_GAME:
    def __init__(self,num_agents) -> None:
        self.bots = [Agent(
            (randaxis == 1) * Vector2(randpos, cell_number - 2) + (randaxis == 2) * Vector2(cell_number - 2, randpos) + (randaxis == 3) * Vector2(randpos, 1) + (randaxis == 4) * Vector2(1, randpos),
            (randaxis == 1) * Vector2(randpos, cell_number - 1) + (randaxis == 2) * Vector2(cell_number - 1, randpos) + (randaxis == 3) * Vector2(randpos, 0) + (randaxis == 4) * Vector2(0, randpos)
        ) for _ in range(num_agents) for randpos, randaxis in [(randint(0, cell_number - 1), randint(1, 4))]]
        self.num_agents = num_agents
        self.end_pts = [END_PNT(Vector2(randint(0,cell_number-1),randint(0,cell_number-1))) for _ in range(num_agents)]
        self.update_state()
        # print(f'game vector: {self.game_vector}')

    '''
    FIXME:  fix this method to udpate the state each bot in the bots attribute
    and use the new collision method to check for collisions
    '''
    def update_state(self):
        self.entities = [self.bots,self.end_pts]
        self.collision = np.zeros((self.num_agents,self.num_agents))
        for i, j in combinations(range(len(self.bots)),2):
            if self.bots[j].collision(self.bots[i]):
                self.collision[j][i] = 1
        # return self.collision
        self.dones = np.zeros(self.num_agents)
        for i, bot in enumerate(self.bots):
            bot.update_state(self)
            if bot.collision(self.end_pts[i]):
                self.dones[i] = 1
        # print(self.collision)

    '''
    TODO: return calculated rewards, dones, scores
    check for compatibility with the updated action
    '''
    def step(self,bots_action: list[Action]):
        boundaries = 0
        for idx, bot in enumerate(self.bots):
            try:
                bot.move_agent(bots_action[idx])
            except IndexError as e:
                bot.move_agent(Action.NO_ACTION)
                boundaries += 1
        if boundaries == len(self.bots):
            print('boundary')
            pygame.time.delay(1000)
            main_game.__init__(self.num_agents)
        else:
            self.update_state()
        # return rewards, dones, scores

    def draw_elements(self):
        for idx, bot in enumerate(self.bots):
            bot.draw_agent()
            self.end_pts[idx].draw_end_pnt()

    '''
    FIXME: make sure this resets the state correctly
    '''
    def reset(self):
        self.bots = [Agent(
            (randaxis == 1) * Vector2(randpos, cell_number - 2) + (randaxis == 2) * Vector2(cell_number - 2, randpos) + (randaxis == 3) * Vector2(randpos, 1) + (randaxis == 4) * Vector2(1, randpos),
            (randaxis == 1) * Vector2(randpos, cell_number - 1) + (randaxis == 2) * Vector2(cell_number - 1, randpos) + (randaxis == 3) * Vector2(randpos, 0) + (randaxis == 4) * Vector2(0, randpos)
        ) for _ in range(num_agents) for randpos, randaxis in [(randint(0, cell_number - 1), randint(1, 4))]]
        self.num_agents = num_agents
        self.end_pts = [END_PNT(randint(0,cell_number-1),randint(0,cell_number-1)) for _ in range(num_agents)]
        self.update_state()    

    def train(self):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = np.zeros(num_agents)
        while  True:
            # get old state
            states_old = [bot.state for bot in self.bots]

            # get move
            final_moves = [bot.get_action(states_old[i]) for i, bot in enumerate(self.bots)]

            # perform move and get new state
            rewards, dones, scores = self.step(final_moves)
            states_new = [bot.state for bot in self.bots]

            # short term memory
            for i, bot in enumerate(self.bots): bot.train_short_memory(states_old[i],final_moves[i],rewards[i],states_new[i],dones[i])

            # remember
            for i, bot in enumerate(self.bots): bot.remember(states_old[i],final_moves[i],rewards[i],states_new[i],dones[i])

            if np.sum(dones) == num_agents:
                self.reset()
                for bot in self.bots: bot.n_games += 1
                for bot in self.bots: bot.train_long_memory()

                for i, score in scores:
                    if score > record[i]:
                        record[i] = score

                print(f'Game: {self.bots[0].n_games}, Scores: {scores}')

if __name__ == '__main__':
        
    pygame.init()

    # will create a square block w = 40 px, h = 40 px
    cell_size = 40
    cell_number = 20
    num_agents = 2
    screen = pygame.display.set_mode((cell_number * cell_size,cell_number * cell_size))
    clock = pygame.time.Clock()

    SCREEN_UPDATE = pygame.USEREVENT
    # COLLISION = pygame.USEREVENT+1
    # TERMINAL_STATE = pygame.USEREVENT+2
    pygame.time.set_timer(SCREEN_UPDATE,150)
    # pygame.time.set_timer(COLLISION, 150)
    # pygame.time.set_timer(TERMINAL_STATE,150)

    main_game = MAIN_GAME(num_agents)
    # print(main_game.collision)


    while True:

        action = [Action.RIGHT,Action.LEFT]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == SCREEN_UPDATE:
                main_game.step(action)
                if main_game.collision:
                    print('collision')
                    pygame.time.delay(1000)
                    main_game.reset(main_game.num_agents)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print('reset')
                    main_game.reset(main_game.num_agents)

        screen.fill((175,215,70))
        main_game.draw_elements()
        pygame.display.update()
        clock.tick(60)
