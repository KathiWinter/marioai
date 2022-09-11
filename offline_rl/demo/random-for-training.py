import os
import random
from random import randint
import argparse
import time

import gym
import numpy as np
from os.path import exists

import sys
#current_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(current_dir, ".."))

from gym_marioai import levels


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--level", type=str)
parser.add_argument("-s", "--seed", default=0)
args = parser.parse_args()

def get_level(input=args.level):
    if input == "easyLevel":
        return levels.easy_level
    elif input == "cliffLevel":
        return levels.cliff_level
    elif input == "earlyCliffLevel":
        return levels.early_cliff_level
    elif input == "coinLevel":
        return levels.coin_level
    elif input == "enemyLevel":
        return levels.enemy_level
    elif input == "flatLevel":
        return levels.flat_level
    elif input == "hardLevel":
        return levels.hard_level
    elif input == "oneCliffLevel":
        return levels.one_cliff_level
    else:
        return None


def get_seed(input=args.seed):
    if input == "random":
        seed = randint(0,1000)
        return seed
    else:
        return int(input)
'''
#random action for cliff level
def get_random_action():
    rand = random.randint(0, 51)
    if rand >= 0 and rand < 40:
        return env.SPEED_RIGHT
    elif rand >= 40 and rand < 50:
        return env.SPEED_JUMP_RIGHT
    else:
        return env.NOTHING
'''
'''
#without LEFT action
def get_random_action():
    rand = random.randint(0, 3)
    if rand == 0:
        return env.JUMP
    elif rand == 1:
        return env.SPEED_RIGHT
    elif rand == 2:
        return env.SPEED_JUMP_RIGHT
    else:
        return env.NOTHING
'''



#include LEFT action
def get_random_action():
    rand = random.randint(0, 45)
    if rand >= 0 and rand < 3 :
        return env.SPEED_LEFT
    elif rand >= 3 and rand < 6:
        return env.SPEED_JUMP_LEFT
    elif rand >= 6 and rand < 10:
        return env.DOWN
    elif rand >= 10 and rand < 20:
        return env.SPEED_RIGHT
    elif rand >= 20 and rand < 30:
        return env.SPEED_JUMP_RIGHT
    elif rand >= 30 and rand < 40:
        return env.JUMP
    else:
        return env.NOTHING

    
all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)
    
env = gym.make('Marioai-v0', render=True,
            compact_observation=False, #this must stay false for proper saving in dataset
            enabled_actions=all_actions,
            rf_width=20, rf_height=10)

level = get_level()

while True:
    #new episode
    if level != None:
        level_str = str(args.level)
        state = env.reset(level_path=level)
    else:
        seed = get_seed()
        level_str = str(seed)
        print("Playing on seed " + level_str)
        state = env.reset(seed=seed)
        
    done = False
    total_reward = 0
    
    #initialize data arrays with initial states for each episode
    observations = [state]
    actions = [12] #nothing
    rewards = [0]
    terminals = [done]

    while not done:
        action = get_random_action()
        next_state, reward, done, info = env.step(action)
        
        observations.append(next_state)
        actions.append(action)
        rewards.append(reward)
        terminals.append(done)
     
        total_reward += reward
        
    #create Markov-Decision-Process Dataset from collected episode
    datafile_name = "random_generated" + "_" + level_str + "_" + "reward" + str(int(total_reward)) + "_" + str(round(time.time())) 
    
    datapath = os.path.join("../data", datafile_name)
    
    data = np.savez(datapath, observations=observations, actions=actions, rewards=rewards, terminals=terminals)

    print(f'finished episode, total_reward: {total_reward}')