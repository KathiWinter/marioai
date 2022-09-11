import os
import glob
import numpy as np
from os.path import exists
from d3rlpy.dataset import MDPDataset
from constants import DATAPATH




episode_start = np.load('../data/user0_0_reward162_1662924687.npz')

observation_start = episode_start['observations']
actions_start = episode_start['actions']
reward_start = episode_start['rewards']
terminals_start = episode_start['terminals']

if not exists(DATAPATH):
    dataset = MDPDataset(np.array(observation_start), np.array(actions_start), np.array(reward_start), np.array(terminals_start))   
else:
    dataset = MDPDataset.load(DATAPATH)
    dataset.append(np.array(observation_start), np.array(actions_start), np.array(reward_start), np.array(terminals_start))
dataset.dump(DATAPATH)

counter = 0
#create Markov-Decision-Process Dataset from collected data or append existing MDP Dataset
for file in glob.glob(os.path.join('../data/', '*.npz')):
    episode_data = np.load(file)
        
    observations = episode_data["observations"]
    actions = episode_data["actions"]
    rewards = episode_data["rewards"]
    terminals = episode_data["terminals"]
        
    if not exists(DATAPATH):
        dataset = MDPDataset(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))   
    else:
        dataset = MDPDataset.load(DATAPATH)
        dataset.append(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))
            
    counter += 1    
    print(counter)

#save dataset in file path for episode
dataset.dump(DATAPATH)
