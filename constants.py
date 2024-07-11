from enum import Enum
SNAKE_1_Q_TABLE = "s1_qtble.npy"
SNAKE_2_Q_TABLE = "s2_qtble.npy"

WIDTH = 500
HEIGHT = 500
NUM_SATAES = 256 # 8 (number of dangers) * 4  (number of directions) * 8 numbers of food directions
NUM_ACTIONS = 4
EPSILON = 1
EPSILON_MEAN = 0.01
EPSILON_DECENT = 0.999
DISCOUNT_FACTOR = 0.2
LEARNING_RATE = 0.1
ROWS = 20
GETTING_PRIZE = 100
HITTING_ITSELF = -5
OUT_BOARD_PUNISH = -10
EAT_OTHER_SNAKE = -10
EAT_OTHER_SNAKE_AND_LONGER = 5
EAT_OTHER_SNAKE_AND_EQUAL = 5
EAT_OTHER_SNAKE_AND_SHORTER = -10


class DIRECTION(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

class FOOD_DIRECTION(Enum):
    UP = 0
    UP_RIGHT = 1
    RIGHT = 2
    RIGHT_DOWN = 3
    DOWN = 4
    DOWN_LEFT = 5
    LEFT = 6
    LEFT_UP = 7
    