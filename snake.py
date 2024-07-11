from cube import Cube
from constants import *
from utility import *
import pickle 
import random
import numpy as np
import matplotlib.pyplot as plt

class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        # pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            with open(file_name, 'rb') as f: 
                self.q_table = pickle.load(f)
        except:
            self.q_table = dict()
            # TODO: Initialize Q-table (done)

        self.lr = LEARNING_RATE 
        self.discount_factor =  DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.hist = [] 
        self.hist_reward = [] 
        self.file_name = file_name


# TODO: how optimal policy should be definded?
# TODO: how update the Q-table?


    def get_optimal_policy(self, state):
        return np.argmax(self.q_table[state])

    def make_action(self, state):
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        # TODO: Update Q-table(done)
        if state not in self.q_table: #if state not in qtable feel it with 0 a
            self.q_table[state] = np.zeros(5)
            self.q_table[state][state[1]] = 1
            
        if next_state not in self.q_table: #if next state not in qtable feel it with 0 a
            self.q_table[next_state] = np.zeros(5)
            self.q_table[next_state][next_state[1]] = 1
            
        self.q_table[state][action] = self.q_table[state][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(self.q_table[next_state]) - self.q_table[state][action])
        if self.epsilon > EPSILON_MEAN :
            self.epsilon = self.epsilon * EPSILON_DECENT

    def move(self, snack, other_snake):
        
        state = self.get_state(snack, other_snake)
        action = self.make_action(state)
        if action == DIRECTION.LEFT.value: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == DIRECTION.RIGHT.value: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == DIRECTION.UP.value: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == DIRECTION.DOWN.value: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)
        new_state = self.get_state(snack, other_snake)
        return state, new_state, action
        # TODO: Create new state after moving and other needed values and return them(done)
        
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False
    
    def get_state(self, snack, other_snake):
        
        # States =
        # [danger straight, danger right, danger left,
        # moving direcetion,
        # food direcion]
        
        # Assumed that headPos[0] is x, headPos[1] is y
        
        headPos = self.head.pos
        point_left = headPos[0] - 1
        point_right = headPos[0] + 1
        point_up = headPos[1] - 1
        point_down = headPos[1] + 1
        
        dir_r = 0 
        dir_l= 0 
        dir_u = 0 
        dir_d = 0
        moving_dir = 0
        
        if self.dirnx == 1:
            dir_r = 1
            moving_dir = 1
        elif self.dirnx == -1:
            dir_l = 1 
            moving_dir = 0
        elif self.dirny == -1:        
            dir_u = 1
            moving_dir = 2
        else:    
            dir_d = 1
            moving_dir = 3
        
        # state = None # TODO: Create state
        state = [
            # danger straight
            int((dir_l and self.is_danger(point_left, other_snake)) or
            (dir_r and self.is_danger(point_right, other_snake)) or
            (dir_u and self.is_danger(point_up, other_snake)) or
            (dir_d and self.is_danger(point_down, other_snake))),
            # danger right
            int((dir_l and self.is_danger(point_up, other_snake)) or
            (dir_r and self.is_danger(point_down, other_snake)) or
            (dir_u and self.is_danger(point_right, other_snake)) or
            (dir_d and self.is_danger(point_left, other_snake))),
            # danger left
            int((dir_l and self.is_danger(point_down, other_snake)) or
            (dir_r and self.is_danger(point_up, other_snake)) or
            (dir_u and self.is_danger(point_left, other_snake)) or
            (dir_d and self.is_danger(point_right, other_snake))),
            # Move direction
            moving_dir,
            # Food location
            self.location_of_snack(snack)
        ]
        return tuple(state)  
     
    def is_collision(self, point):
        if point >= ROWS - 1 or point < 1 or point >= ROWS - 1 or point < 1:
            return True
        return False
    
    def is_other_snake(self, point, other_snake):
        if point in list(map(lambda z: z.pos, other_snake.body)) and point != other_snake.head.pos and len(self.body) < len(other_snake.body):
            return True
        return False
    
    def is_danger(self, point, other_snake):
        if (self.is_collision(point) | self.is_other_snake(point, other_snake)):
            return True
        return False    
    
    def location_of_snack(self, snack):
        
        food_direcion = 0
        
        if (snack.pos[1] < self.head.pos[1]): # snack is up side of the snake
            food_direcion = FOOD_DIRECTION.UP.value
        elif (snack.pos[1] < self.head.pos[1] and snack.pos[0] > self.head.pos[0]): # snack is up-right side of the snake
            food_direcion = FOOD_DIRECTION.UP_RIGHT.value     
        elif (snack.pos[0] > self.head.pos[0]): # snack is right side of the snake
            food_direcion = FOOD_DIRECTION.RIGHT.value    
        elif (snack.pos[1] > self.head.pos[1] and snack.pos[0] > self.head.pos[0]): # snack is right-down side of the snake
            food_direcion = FOOD_DIRECTION.RIGHT_DOWN.value
        elif (snack.pos[1] > self.head.pos[1]): # snack is down side of the snake
            food_direcion = FOOD_DIRECTION.DOWN.value  
        elif (snack.pos[1] > self.head.pos[1] and snack.pos[0] < self.head.pos[0]): # snack is down-left side of the snake
            food_direcion = FOOD_DIRECTION.DOWN_LEFT.value       
        elif (snack.pos[0] < self.head.pos[0]): # snack is left side of the snake
            food_direcion = FOOD_DIRECTION.LEFT.value
        elif (snack.pos[1] < self.head.pos[1] and snack.pos[0] < self.head.pos[0]): # snack is up-left side of the snake
            food_direcion = FOOD_DIRECTION.LEFT_UP.value         
        return food_direcion    
                        
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        
        if self.check_out_of_board():
            # TODO: Punish the snake for getting out of the board(done)
            reward =+ OUT_BOARD_PUNISH
            win_other = True
            reset(self, other_snake, win_other)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward =+ GETTING_PRIZE
            # TODO: Reward the snake for eating(done)
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # TODO: Punish the snake for hitting itself (done)
            reward =+ HITTING_ITSELF
            win_other = True
            reset(self, other_snake, win_other)
            
            
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                # TODO: Punish the snake for hitting the other snake(done)
                reward =+ EAT_OTHER_SNAKE
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    reward =+ EAT_OTHER_SNAKE_AND_LONGER
                    # TODO: Reward the snake for hitting the head of the other snake and being longer(done)
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    reward =+ EAT_OTHER_SNAKE_AND_EQUAL
                    # TODO: No winner(done)
                    pass
                else:
                    reward =+ EAT_OTHER_SNAKE_AND_SHORTER
                    # TODO: Punish the snake for hitting the head of the other snake and being shorter(done)
                    win_other = True
            self.hist_reward.append(reward)        
            reset(self, other_snake, win_other)
            
        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        if len(self.hist_reward) > 2 and 's1' in self.file_name:
            self.hist.append(np.mean(self.hist_reward))
            self.hist_reward = []
            
            if len(self.hist) % 10 == 9:
                plt.plot(self.hist)
                plt.savefig(self.file_name[:3] + 'img')

            if len(self.hist) % 100 == 99:
                self.lr *= 0.95
                self.epsilon *= 0.92
                print(self.lr, self.epsilon)

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        # This function save the q table
        np.save(file_name, self.q_table)
        