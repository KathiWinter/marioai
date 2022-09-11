import gym
import gym_marioai
import numpy as np
from os.path import exists
from gym_marioai import levels
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH, POLICY
import torch
import csv
from cdqn import CDQN

dataset = MDPDataset.load(DATAPATH)

### Evaluation of our implemented CDQN algorithm based on the d3rlpy DQN with logging to CSV file ###
cdqn = CDQN()


#use this instead of dqn.fit when dqn.fit() has already been run
cdqn.build_with_dataset(dataset)

cdqn.load_model('../evaluations\group_contest\super_run_1000\model_1001.pt')


all_actions = (0,1,2,3,4,5,6,7,8,9,10,11,12)

env = gym.make('Marioai-v0', render=False,
               compact_observation=False, #this must stay false for proper saving in dataset
               enabled_actions=all_actions,
               rf_width=20, rf_height=10)

header = ['seed', 'total_reward', 'win', 'steps']
with open('example.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for seed in range(1001,1051):
        observation = env.reset(seed=seed)
        done = False
        total_reward = 0
            
        while not done:
            a = np.random.randint(0,10)
            ## change the comments for stochastic actions ##
            '''
            if a < 1:
                action = np.random.randint(0,13)
            else:
                action = cdqn.predict([observation])[0]
            '''
            action = cdqn.predict([observation])[0]
            observation, reward, done, info = env.step(action)
            
            if (reward == 25): #cliff 
                print("jumped over cliff. Change reward.")
                reward = 0
            
            total_reward += reward
            
        ls = [seed, total_reward, info["win"], info["steps"]]
        with open('example.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ls)