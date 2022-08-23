import keyboard

import gym
import gym_marioai
import numpy as np
from gym_marioai import levels
from d3rlpy.dataset import MDPDataset


all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)

env = gym.make('Marioai-v0', render=True,
               level_path=levels.coin_level,
               compact_observation=True,
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)

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

#initialize data arrays
observations = np.empty(shape=(0))
actions = np.empty(shape=(0))
rewards = np.empty(shape=(0))
terminals = np.empty(shape=(0))

#play loop: execute actions from keyboard input in the environment
while True:
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = get_action()
        print('action', action)
        next_state, reward, done, info = env.step(action)
        
        np.append(observations, next_state)
        np.append(actions, action)
        np.append(rewards, reward)
        np.append(terminals, done)
     
        total_reward += reward

    print(f'finished episode, total_reward: {total_reward}')
    
    #create Markov-Decision-Process Dataset from collected data
    dataset = MDPDataset(observations, actions, rewards, terminals)    
    
    #save dataset as file
    dataset.dump('dataset.h5')

#we should introduce a proper finish-criterion 
print('finished demo')



