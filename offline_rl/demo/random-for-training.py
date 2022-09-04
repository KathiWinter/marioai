import os
import random
from random import randint
import argparse
import time

import gym
import gym_marioai
import numpy as np
from os.path import exists
from gym_marioai import levels


parser = argparse.ArgumentParser()
parser.add_argument("-u", "--user", type=int, default=0) #0=testuser
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

def get_random_action():
    rand = random.randint(0, 12)
    if rand == 0:
        return env.LEFT
    elif rand == 1:
        return env.RIGHT
    elif rand == 2:
        return env.UP
    elif rand == 3:
        return env.DOWN
    elif rand == 4:
        return env.JUMP
    elif rand == 5:
        return env.SPEED_JUMP
    elif rand == 6:
        return env.SPEED_RIGHT
    elif rand == 7:
        return env.SPEED_LEFT
    elif rand == 8:
        return env.JUMP_RIGHT
    elif rand == 9:
        return env.JUMP_LEFT
    elif rand == 10:
        return env.SPEED_JUMP_RIGHT
    elif rand == 11:
        return env.SPEED_JUMP_LEFT
    elif rand == 12:
        return env.NOTHING


if __name__ == '__main__':
    
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
        datapath = os.path.join("../data/random_Generated_Episode", datafile_name)
    
        data = np.savez(datapath, observations=observations, actions=actions, rewards=rewards, terminals=terminals)

        print(f'finished episode, total_reward: {total_reward}')













if __name__ == '__main__':

    # adjust the reward settings like so:
    reward_settings = gym_marioai.RewardSettings(dead=-10000, timestep=0)

    env = gym.make('Marioai-v1', render=True,
                   reward_settings=reward_settings,
                   level_path=levels.cliff_level,
                   # compact_observation=True,
                   # trace_length=3,
                   rf_width=7, rf_height=5
                   )

    for e in range(100):
        s = env.reset()
        done = False
        total_reward = 0

        while not done:
            a = env.JUMP_RIGHT if random.randint(0,1) % 2 == 0 else env.SPEED_RIGHT
            s, r, done, info = env.step(a)
            total_reward += r

        print(f'finished episode {e}, total_reward: {total_reward}')

    print('finished demo')