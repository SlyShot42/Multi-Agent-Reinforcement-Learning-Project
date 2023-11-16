import pygame, sys
from pygame.math import Vector2
from random import randint
import random
from enum import Enum
import numpy as np
from collections import deque
from itertools import combinations
import torch

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Action(Enum):
    RIGHT =  Vector2(1,0)
    LEFT = Vector2(-1,0)
    UP = Vector2(0,-1)
    DOWN = Vector2(0,1)
    NOTHING = True

class STATE_VECTOR:
    def __init__(self,pts: list[Vector2] = []) -> None:
        self.vector = np.zeros((cell_number,cell_number))
        for pt in pts:
            self.vector[int(pt.x),int(pt.y)] = 1
    
    def __add__(self,other):
        if isinstance(other,STATE_VECTOR):
            zero = STATE_VECTOR()
            zero.vector = self.vector + other.vector
            return zero
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")
        
    def __sub__(self,other):
        if isinstance(other,STATE_VECTOR):
            zero = STATE_VECTOR()
            zero.vector = self.vector - other.vector
            return zero
        else:
            raise TypeError(f"Unsupported type for substraction: {type(other)}")
        
    def __repr__(self):
        return f'MyClass({self.vector})'

    def __str__(self):
        return f'MyClass object with value: {self.vector}'

class Agent:
    def __init__(self,front,back) -> None:
        self.front = front
        self.back = back
        self.body = [self.front,self.back]
        self.obj_vector =  STATE_VECTOR(self.body)
        self.direction = Action.NOTHING
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = None # TODO
        self.trainer = None # TODO


    def get_state(self,game):
        """
        TODO: edit this method so that agent's state is instantiated correctly
        in that it extracts the current game data to create the current state of the
        agent.
        states:
            danger straight, danger right, danger left,
            direction left, direction right, direction up, direction down, direction none,
            end point left, end point right, end point up, end point down
        """
        self.state = []
        for idx, bot in enumerate(game.bots):
            # danger straight
            straight = STATE_VECTOR(bot.front+bot.direction.value.rotation(0))
            for i, other_bot in enumerate(self.bots):
                if i != idx:
                    straight += other_bot.obj_vector
            danger_straight = np.any(straight.vector > 1) 
            self.state.append(danger_straight)

            # danger right
            right = STATE_VECTOR(bot.front+bot.direction.value.rotation(-90))
            for i, other_bot in enumerate(self.bots):
                if i != idx:
                    right += other_bot.obj_vector
            danger_right = np.any(right.vector > 1) 
            self.state.append(danger_right)

            # danger left
            left = STATE_VECTOR(bot.front+bot.direction.value.rotation(90))
            for i, other_bot in enumerate(self.bots):
                if i != idx:
                    left += other_bot.obj_vector
            danger_left = np.any(left.vector > 1) 
            self.state.append(danger_left)

            self.state.append(bot.direction == Action.LEFT) # direction left
            self.state.append(bot.direction == Action.RIGHT) # direction right
            self.state.append(bot.direction == Action.UP) # direction up
            self.state.append(bot.direction == Action.DOWN) # direction down
            self.state.append(bot.direction == Action.NOTHING) # direction none

            self.state.append(bot.front.x > self.end_pts[idx].x) # end point left 
            self.state.append(bot.front.x < self.end_pts[idx].x) # end point right
            self.state.append(bot.front.y > self.end_pts[idx].y) # end point up
            self.state.append(bot.front.y < self.end_pts[idx].y) # end point down


    def draw_agent(self):
        for idx, block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect  = pygame.Rect(x_pos,y_pos,cell_size,cell_size)
            pygame.draw.rect(screen,(idx == 0) * (56,245,237) + (idx == 1) * (126,166,114),block_rect)

    def move_agent(self,action: Action):
        if action == Action.NOTHING:
            return
        body_copy = self.body[:]
        self.back = self.front
        self.front = body_copy[0]+action.value
        body_copy.insert(0,self.front)
        self.body = body_copy[:-1]
        self.obj_vector = STATE_VECTOR(self.body)
        self.direction = action.value

    def get_action(self, state) -> list[Action]:
        self.epsilon = 80 - self.n_games
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

    def train(self,game):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        while  True:
            # get old state
            state_old = self.get_state(game)

            # get move
            final_move = self.get_action(state_old)

            # reward move and get a new state
            reward, done, score = game.step(final_move)
            state_new = self.get_state(game)

            # short term memory
            self.train_short_memory(state_old,final_move,reward,state_new,done)

            # remember
            self.remember(state_old,final_move,reward,state_new,done)

            if done:
                game.reset()
                self.n_games += 1
                self.train_long_memory()

                if score > record:
                    record = score

                print(f'Game: {self.n_games}, Score: {score}')

class END_PNT:
    def __init__(self,x,y) -> None:
        self.x = randint(0,cell_number-1)
        self.y = randint(0,cell_number-1)
        self.pos = Vector2(self.x,self.y)
        self.obj_vector = STATE_VECTOR([self.pos])

    def draw_end_pnt(self):
        terminal_rect = pygame.Rect(int(self.pos.x*cell_size),int(self.pos.y*cell_size),cell_size,cell_size)
        pygame.draw.rect(screen,(255,0,162),terminal_rect)
        
class MAIN_GAME:
    def __init__(self,num_agents) -> None:
        self.bots = [Agent(
            (randaxis == 1) * Vector2(randpos, cell_number - 2) + (randaxis == 2) * Vector2(cell_number - 2, randpos) + (randaxis == 3) * Vector2(randpos, 1) + (randaxis == 4) * Vector2(1, randpos),
            (randaxis == 1) * Vector2(randpos, cell_number - 1) + (randaxis == 2) * Vector2(cell_number - 1, randpos) + (randaxis == 3) * Vector2(randpos, 0) + (randaxis == 4) * Vector2(0, randpos)
        ) for _ in range(num_agents) for randpos, randaxis in [(randint(0, cell_number - 1), randint(1, 4))]]
        self.num_agents = num_agents
        # self.bots = [Agent(Vector2(1,5),Vector2(0,5)),Agent(Vector2(cell_number-2,5),Vector2(cell_number-1,5))]
        self.end_pts = [END_PNT(randint(0,cell_number-1),randint(0,cell_number-1)) for _ in range(num_agents)]
        self.update_state()
        

    def update_state(self):
        self.entities = [self.bots,self.end_pts]
        self.game_vector = STATE_VECTOR()
        for entity_type in self.entities:
            for entity in entity_type:
                self.game_vector += entity.obj_vector
        self.collision = np.any(self.game_vector.vector > 1)
        # print(self.collision)


        # print(f'game vector: {self.game_vector}')

    def step(self,bots_action: list[Action]):
        boundaries = 0
        for idx, bot in enumerate(self.bots):
            try:
                bot.move_agent(bots_action[idx])
            except IndexError as e:
                bot.move_agent(Action.NOTHING)
                boundaries += 1
        if boundaries == len(self.bots):
            print('boundary')
            pygame.time.delay(1000)
            main_game.__init__(self.num_agents)
        else:
            self.update_state()
        return reward, done, score

    def draw_elements(self):
        for idx, bot in enumerate(self.bots):
            bot.draw_agent()
            self.end_pts[idx].draw_end_pnt()

    def reset(self):
        self.bots = [Agent(
            (randaxis == 1) * Vector2(randpos, cell_number - 2) + (randaxis == 2) * Vector2(cell_number - 2, randpos) + (randaxis == 3) * Vector2(randpos, 1) + (randaxis == 4) * Vector2(1, randpos),
            (randaxis == 1) * Vector2(randpos, cell_number - 1) + (randaxis == 2) * Vector2(cell_number - 1, randpos) + (randaxis == 3) * Vector2(randpos, 0) + (randaxis == 4) * Vector2(0, randpos)
        ) for _ in range(num_agents) for randpos, randaxis in [(randint(0, cell_number - 1), randint(1, 4))]]
        self.num_agents = num_agents
        # self.bots = [Agent(Vector2(1,5),Vector2(0,5)),Agent(Vector2(cell_number-2,5),Vector2(cell_number-1,5))]
        self.end_pts = [END_PNT(randint(0,cell_number-1),randint(0,cell_number-1)) for _ in range(num_agents)]
        self.update_state()    



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
print(main_game.collision)


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
