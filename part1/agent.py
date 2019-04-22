import numpy as np
import utils
import random

import time ##
class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne            # used in exploration function
        print("Ne", Ne)
        print("gamma", gamma)
        self.C = C              # constant for learning rate
        self.gamma = gamma      # discount

        # Create the Q and N Table to work with
        # At the beginning, it should be all zeros for Q-table and N-table
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.points = 0
        self.s = None
        self.a = None #?

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
    def discretize_state(self, state):
        GRID_SIZE = utils.GRID_SIZE
        head_x, head_y, snake_body, food_x, food_y = state
        ### discretize the state to the state space defined on the webpage
        adjoining_wall_x = 0
        adjoining_wall_y = 0
        if (head_x == GRID_SIZE) :
            adjoining_wall_x = 1
        elif (head_x == utils.DISPLAY_SIZE- 2 * GRID_SIZE):
            adjoining_wall_x = 2
        if (head_y == GRID_SIZE) :
            adjoining_wall_y = 1
        elif (head_y == utils.DISPLAY_SIZE- 2 * GRID_SIZE):
            adjoining_wall_y = 2
        food_dir_x = 0
        food_dir_y = 0
        # if (head_x < 2*GRID_SIZE) :
        #     adjoining_wall_x = 1 # wall on the left
        # elif (head_x + 2*utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE) :
        #     adjoining_wall_x = 2 # wall on the right
        # else :
        #     adjoining_wall_x = 0 # not close to wall on left or right
        # if (head_y < 2*GRID_SIZE) :
        #     adjoining_wall_y = 1 # wall on the top
        # elif (head_y + 2*utils.GRID_SIZE > utils.DISPLAY_SIZE-utils.GRID_SIZE) :
        #     adjoining_wall_y = 2 # wall on the bottom
        # else :
        #     adjoining_wall_y = 0 # not close to wall on top or bottom
        if (food_x == head_x) :
            food_dir_x = 0
        elif (food_x < head_x) :
            food_dir_x = 1
        else :
            food_dir_x = 2
        if (food_y == head_y) :
            food_dir_y = 0
        elif (food_y < head_y) :
            food_dir_y = 1
        else :
            food_dir_y = 2
        adjoining_body_left = 0
        adjoining_body_right = 0
        adjoining_body_top = 0
        adjoining_body_bot = 0
        for seg in snake_body:
            if head_x - GRID_SIZE == seg[0] and head_y == seg[1]:
                adjoining_body_left = 1
            if head_x == seg[0] and head_y - GRID_SIZE == seg[1]:
                adjoining_body_top = 1
            if head_x + GRID_SIZE == seg[0] and head_y == seg[1]:
                adjoining_body_right = 1
            if head_x == seg[0] and head_y + GRID_SIZE == seg[1]:
                adjoining_body_bot = 1
        return (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bot, adjoining_body_left, adjoining_body_right)
    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        R_plus = 1
        # print("agent line 51: start of act", state, points, dead)
        # print("52", head_x, head_y, snake_body, food_x, food_y)
        d1, d2, d3, d4, d5, d6, d7, d8 = self.discretize_state(state)
        # print("State",state)
        # print("adjoining_wall_x",d1, "adjoining_wall_y",d2, "food_dir_x",d3, "food_dir_y", d4, d5, d6, d7, d8)
        # print("points", points)

        ## STEP 1 update Q-Table
        # Q(s,a)←Q(s,a)+α(R(s)+γmax_a′Q(s′,a′)−Q(s,a))
        # if first time (self.s or self.a is None), skip
        if (self.s != None and self.a !=  None):
            s1, s2, s3, s4, s5, s6, s7, s8 = self.s
            max_q = -np.inf
            for action in self.actions:
                if self.Q[d1][d2][d3][d4][d5][d6][d7][d8][action] > max_q:
                    max_q = self.Q[d1][d2][d3][d4][d5][d6][d7][d8][action]
            # alpha = C/(C+N(s,a))
            alpha = self.C*1.0 / (self.C + self.N[s1][s2][s3][s4][s5][s6][s7][s8][self.a])
            rewards = -0.1
            if dead :
                rewards = -1
            if points- self.points > 0:
                rewards = 1
            self.Q[s1][s2][s3][s4][s5][s6][s7][s8][self.a] += alpha * (rewards + self.gamma * max_q - self.Q[s1][s2][s3][s4][s5][s6][s7][s8][self.a])
        ## STEP 2 check if dead
        if dead :
            self.reset()
            return 0
        else :
            self.points = points
            self.s = (d1, d2, d3, d4, d5, d6, d7, d8)

        ## STEP 3 find the best action
        max_f = -np.inf
        best_action = 0
        for action in self.actions:
            n = self.N[d1][d2][d3][d4][d5][d6][d7][d8][action]
            if n < self.Ne :
                # current f-value is R_plus
                if R_plus >= max_f:
                    max_f = R_plus
                    best_action = action
            else:
                q = self.Q[d1][d2][d3][d4][d5][d6][d7][d8][action]
                if q >= max_f:
                    max_f = q
                    best_action = action
        # print("best_action", best_action)
        self.a = best_action
        ## STEP 4 update N-table
        self.N[d1][d2][d3][d4][d5][d6][d7][d8][best_action] += 1


        return best_action
