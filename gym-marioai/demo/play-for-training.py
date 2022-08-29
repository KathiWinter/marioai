import os
import keyboard

import gym
import gym_marioai
import numpy as np
from os.path import exists
from gym_marioai import levels
from d3rlpy.dataset import MDPDataset


all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)

env = gym.make('Marioai-v0', render=True,
               level_path=levels.coin_level,
               compact_observation=False, #this must stay false for proper saving in dataset
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)

#TODO: Should we save a dataset for each player or for each episode? Or just join all datasets?
datapath = os.path.join("dataset.h5")

def get_action():
    if keyboard.is_pressed('up'):
        return env.JUMP
    elif keyboard.is_pressed('right'):
        return env.SPEED_RIGHT
    elif keyboard.is_pressed('left'):
        return env.SPEED_LEFT
    elif keyboard.is_pressed('down'):
        return env.DOWN
    

    elif keyboard.is_pressed('d'):
        return env.SPEED_JUMP_RIGHT
    elif keyboard.is_pressed('a'):
        return env.SPEED_JUMP_LEFT
    
    else:
        return env.NOTHING

#play loop: execute actions from keyboard input in the environment
while True:
    #new episode
    state = env.reset()
    done = False
    total_reward = 0
    
    #initialize data arrays with initial states for each episode
    observations = [state]
    actions = [12] #nothing
    rewards = [0]
    terminals = [done]

    while not done:
        action = get_action()
        print('action', action)
        next_state, reward, done, info = env.step(action)
        
        observations.append(next_state)
        actions.append(action)
        rewards.append(reward)
        terminals.append(done)
     
        total_reward += reward

    print(f'finished episode, total_reward: {total_reward}')
    
    #create Markov-Decision-Process Dataset from collected data or append existing MDP Dataset
    if not exists(datapath):
        dataset = MDPDataset(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))   
    else:
        dataset = MDPDataset.load(datapath)
        dataset.append(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))
        
    #save dataset in file path for episode
    dataset.dump(datapath)

#TODO: we should introduce a proper finish-criterion 
print('finished demo')



