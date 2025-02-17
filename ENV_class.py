import random
from CONFI import*
from CSV_file import*
import pandas as pd

class CustomEnv:
    def __init__(self, factor_ranges):
        self.factor_ranges = factor_ranges
        self.reset()

    def reset(self):
        # Initialize factors with random starting values
        #test1:random.uniform(*self.factor_ranges[0])
        #test2:0
        self.a = int((MAX_A+MIN_A)/2)
        self.b = int((MAX_B+MIN_B)/2)
        self.c = int((MAX_C+MIN_C)/2)
        self.previous_mean = 0  # Start with no previous mean
        return [self.a, self.b, self.c, self.previous_mean]  # Return initial state

    def step(self,steps_done,action,countNum):
        # Define actions (0-5): 0 and 1 adjust `a`, 2 and 3 adjust `b`, 4 and 5 adjust `c`
        factor_index = action // 2
        adjustment = STEPSIZE if action % 2 == 0 else (0-STEPSIZE)

        # Adjust the selected factor
        if factor_index == 0:
            self.a += adjustment
        elif factor_index == 1:
            self.b += adjustment
        elif factor_index == 2:
            self.c += adjustment

        # Keep values within ranges
        self.a = max(self.factor_ranges[0][0], min(self.a, self.factor_ranges[0][1]))
        self.b = max(self.factor_ranges[1][0], min(self.b, self.factor_ranges[1][1]))
        self.c = max(self.factor_ranges[2][0], min(self.c, self.factor_ranges[2][1]))

        a= self.a
        b= self.b
        c= self.c

        mean = log_factors(FILE_PATH_INTERACT, countNum, step=steps_done, factor_1=a, factor_2=b, factor_3=c, mean= None)
        mean = get_mean_for_step(FILE_PATH_INTERACT,countNum)
        # Calculate the new mean
        #mean = (self.a + self.b + self.c) / 3

        # Calculate reward
        # Reward based on improvement
        if mean > self.previous_mean:
            reward = 1 + (mean - self.previous_mean)  # Reward improvement
        elif mean == self.previous_mean:
            reward = 0.5  # Small reward for stability
        else:
            reward = -1  # Penalty for decrease
        # Additional stability bonus if mean is near the ideal range (25-30)
        #if 8 <= mean <= 10:
        #    reward += 2  # Reward for reaching the target range
        self.previous_mean = mean

        # Define done criteria (e.g., episode ends after a set number of actions or when reaching a target mean)
        done = False
        if mean>0:
            done = True
            #print("a done")
        #else: print ("a not done",self.a, self.b, self.c, mean)
        # Return new state, reward, and done flag
        return [self.a, self.b, self.c, mean], reward, done

'''
    def sample_action(self):
        # Sample a random action (0 through 5)
        return random.randint(0, 5)
'''
